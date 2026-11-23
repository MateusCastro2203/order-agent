# order-agent

Agente que consulta um pedido e só solicita a liberação do envio quando as regras operacionais
são atendidas. **O LLM decide o caminho; o código decide o que é permitido.**

```bash
uv sync
uv run pytest                                   # 113 testes, ~1 s, sem LLM
uv run order-agent "Libera o envio do pedido ORD-0001" --permission shipment:release --planner rules
```

---

## 1. Solução e funcionamento

O agente recebe uma solicitação em linguagem natural ("Libera o envio do pedido ORD-0001"),
decide quais ferramentas chamar e devolve uma resposta baseada **só no que as ferramentas
retornaram**.

```
solicitação ─► plan (LLM) ─► validate ─► execute ─► plan … ─► finalize ─► resposta
                               │  gate      │ timeout/5xx no release
                               └► finalize  └► reconcile ─► escalate (UNKNOWN)
```

- `plan` é o **único** node que chama o LLM: ele propõe a intent e a próxima tool call.
- `validate`, `execute`, `reconcile`, `escalate` e `finalize` são código determinístico.
- Ferramentas simuladas: `get_order`, `get_payment`, `release_shipment`, sobre um sistema
  externo em memória (`tools/system.py`) com falhas injetáveis.
- A resposta final é montada por template a partir de um **outcome** tipado. O texto do
  modelo nunca chega ao usuário.

Outcomes: `RELEASED`, `ORDER_INFO`, `BLOCKED`, `INVALID_REQUEST`, `TOOL_FAILURE`,
`RELEASE_FAILED`, `UNKNOWN`, `ABORTED`. Os três agrupadores carregam um `reason`
(`BLOCKED_PAYMENT`, `UNAUTHORIZED`, `INTENT_MISMATCH`, `NOT_FOUND`, `NO_PROGRESS`…). O modelo
`RunResult` rejeita `released=True` com qualquer outcome que não seja `RELEASED`: uma falha
técnica não tem como virar sucesso.

## 2. Tecnologias e justificativas

| Camada | Escolha | Por quê |
|---|---|---|
| Linguagem | Python 3.12 | Ecossistema de LLM e LangGraph |
| Pacotes | uv | Instalação rápida, lockfile reproduzível |
| Schemas | pydantic v2 | Valida entrada e saída de cada tool; erro vira resultado tipado |
| Orquestração | LangGraph | Estado explícito, roteamento visível e `interrupt()` para a escalada |
| Cliente LLM | httpx direto no `/api/chat` do Ollama | Uma dependência leve, tool calling nativo |
| Modelo | `llama3.1:8b` via Ollama (`LLM_MODEL`) | Gratuito, local, sem credencial, ~4,9 GB |
| Testes | pytest (marker `llm` para o modelo real) | Evals determinísticos por padrão |
| Qualidade | ruff | Lint e formatação |

Três dependências de runtime (`langgraph`, `pydantic`, `httpx`). Sem framework de evals:
os critérios críticos são asserções sobre o trace, e um LLM-as-judge traria
não-determinismo justamente na camada de verificação.

**Quando o LangGraph agrega.** Para um fluxo de três chamadas um loop bastaria. Ele se paga
por dois motivos: o release incerto precisa pausar e retomar de um checkpoint
(`interrupt()`), e o grafo deixa cada caminho, inclusive os de falha, explícito e testável.
As regras ficam fora do grafo: `policy.py` e `tools/` não importam o LangGraph.

**SGLang (proposta, não implementado).** Em produção, com muitas execuções concorrentes, o
SGLang serviria o mesmo modelo com batching contínuo e RadixAttention, que reaproveita o
cache do prefixo comum a toda chamada (system prompt e schemas das tools). Ganho: throughput
e latência por requisição. Custo: operar GPU e um servidor de inferência. O planner fala um
formato próximo ao da OpenAI, então a troca afetaria só o adaptador em `planner/ollama.py`.

## 3. Instalação e execução

Requisitos: [uv](https://docs.astral.sh/uv/). Opcional: [Ollama](https://ollama.com) para o
planner com LLM.

```bash
uv sync

# Testes (sem LLM, sem rede)
uv run pytest

# Agente sem LLM: planner por regras
uv run order-agent "Libera o envio do pedido ORD-0001" --permission shipment:release --planner rules
uv run order-agent "Qual o status do pedido ORD-0002?" --planner rules

# Agente com LLM local
ollama pull llama3.1:8b
uv run order-agent "Libera o envio do pedido ORD-0001" --permission shipment:release
uv run pytest -m llm          # smoke com o modelo real (~80 s)
```

- Permissões vêm de `--permission` (repetível), nunca do texto. Sem
  `--permission shipment:release`, a liberação é bloqueada.
- `--json` imprime o `RunResult` completo; `--quiet` desliga os logs.
- Logs JSON saem no stderr, uma linha por node, com o `run_id`.
- Se o Ollama estiver fora, o CLI sai com código 2 e uma mensagem clara. Não troca de
  planner sozinho.

Pedidos simulados (`tools/fixtures.py`):

| Pedido | Situação |
|---|---|
| ORD-0001 | READY + CONFIRMED (único liberável) |
| ORD-0002 | pagamento PENDING |
| ORD-0003 | pedido CANCELLED |
| ORD-0004 | já SHIPPED |
| ORD-0005 | pedido CREATED |
| ORD-0006 / ORD-0007 | pagamento FAILED / REFUNDED |
| ORD-0404 | não existe |

Limites configuráveis por variável de ambiente: `AGENT_MAX_ITERATIONS`,
`AGENT_MAX_PLANNER_ERRORS`, `AGENT_TOOL_TIMEOUT_S`, `AGENT_LLM_TIMEOUT_S`,
`AGENT_TOTAL_TIMEOUT_S`. Modelo e endpoint: `LLM_MODEL`, `OLLAMA_URL`.

## 4. Decisões técnicas e arquitetura

```
src/order_agent/
├── models.py      schemas imutáveis; outcome + reason validados no modelo
├── policy.py      policy gate (puro, sem LangGraph)
├── intent.py      regra de intent por palavras-chave
├── harness.py     nodes, limites, retry, classificação de erros
├── graph.py       grafo LangGraph (só roteamento) + log por node
├── finalize.py    resposta por template a partir do outcome
├── logs.py        log JSON
├── cli.py         entrada: solicitação, ator, permissões, planner
├── tools/         system.py (sistema simulado), registry.py (execução), fixtures.py
└── planner/       base.py (contrato), scripted.py, rules.py, ollama.py
```

Principais decisões:

1. **O LLM nunca é a única barreira.** O release só é executado se o policy gate aprovar.
2. **A resposta vem do outcome, não do modelo.** Uma falha nunca aparece como sucesso.
3. **Autorização vem do `RunContext`**, passado pelo chamador. "Sou administrador" no texto
   não muda nada (testado com o modelo real).
4. **Release só com a intent do LLM e a regra de palavras-chave de acordo.** Sem isso, um
   LLM que classificasse uma consulta como liberação liberaria o envio. Custo aceito: uma
   liberação escrita com palavras fora da lista é bloqueada, que é o erro seguro.
5. **Leituras ganham 1 retry; o release, nenhum.** Leituras não têm efeito colateral.
6. **Erro 4xx no release é definitivo** (`RELEASE_FAILED`). Timeout, 5xx e resposta
   ilegível são incertos e vão para o reconcile.
7. **Saída das tools validada por schema; valor desconhecido falha fechado.** Um status de
   pagamento novo nunca libera por acidente.
8. **`release_shipment` sempre visível ao LLM.** O bloqueio acontece no gate; esconder a
   tool impediria ver o modelo tentando algo proibido.

## 5. O que não foi implementado e por quê

| Item | Motivo | Como seria |
|---|---|---|
| Retomada pelo CLI após `UNKNOWN` | O enunciado não pede recuperação completa; o que pontua é não repetir nem afirmar sucesso | Checkpointer persistente (SQLite local, Postgres em produção) e `order-agent resume <run_id> --decision released\|not_released`. O node já suporta (`pause_on_uncertain`) |
| Runner de métricas com N execuções | Diferencial; não compensa o essencial | Rodar os cenários N vezes por modelo e agregar o log JSON: taxa de release indevido, falso sucesso, p50/p95, tokens |
| Comparação de modelos e prompts | Depende do runner | `qwen2.5:7b` costuma errar menos JSON em tool calling; trocar via `LLM_MODEL` |
| CI | Cortado por prazo, ver item 7 | GitHub Actions com `uv sync`, `ruff check` e `pytest` |
| Outbox e reconciliação assíncrona | Fora do prazo | Ver item 14 |
| Tracing (Langfuse/LangSmith) | Log JSON local basta para a demo | Exportar os eventos por `run_id` |
| SGLang | Proposta, ver item 2 | — |

## 6. Limitações conhecidas e riscos remanescentes

| Risco | Mitigação atual | Próximo passo |
|---|---|---|
| Release incerto sem idempotência garantida pelo sistema real | Sem retry, reconcile, `UNKNOWN` com verificação manual | Idempotência no sistema externo e outbox |
| TOCTOU: o gate decide com dados lidos antes do release | Janela curta dentro de uma execução; o sistema simulado também recusa com 4xx | Pré-condição validada de forma atômica pelo sistema externo |
| Liberação legítima com palavras fora da lista é bloqueada | Erro seguro, com motivo `INTENT_MISMATCH` na resposta | Confirmação explícita do usuário antes de ações com efeito |
| Modelo 8B erra tool calls e para antes de concluir | Validação de schema, 1 correção, gate determinístico | Comparar modelos e ajustar prompt com o runner |
| Smoke com LLM não é determinístico | Gate de regressão roda só com planners sem LLM | Eval contínuo com dataset versionado |
| Timeout não cancela a thread da tool | Esperado: no release é o cenário que o reconcile trata | Cliente HTTP com cancelamento real |
| Mocks simplificam o sistema real | Falhas injetáveis por cenário | Testes de contrato contra o serviço real |
| Uma solicitação, um pedido | Pedido diferente no meio da execução é recusado | Fan-out por pedido com limite de concorrência |

## 7. Melhorias priorizadas com mais tempo

1. Idempotência garantida pelo sistema externo + outbox com reconciliador assíncrono.
2. Checkpointer persistente e comando `resume` no CLI.
3. CI com ruff e pytest.
4. Runner de métricas e comparação de modelos e prompts.
5. Tracing com Langfuse.
6. Com volume, servir o modelo com SGLang.

## 8. Ferramentas de IA e validação do código

- **Ferramenta:** Claude Code (Claude). Usado para revisar o plano de arquitetura, escrever o
  código e os testes, depurar e redigir este README.
- **Revisão do plano:** antes do código, um agente separado, sem acesso às conclusões
  anteriores, avaliou o plano como avaliador do desafio. Os cortes de escopo vieram dessa
  revisão (sem resume no CLI, um único budget, menos outcomes).
- **Validação:**
  - Todo commit passa por `ruff` e pela suíte completa.
  - **Testes de mutação manuais:** desligar o gate, remover a checagem de intent e dar
    retry ao release, um por vez, quebra os evals correspondentes (2, 1 e 10 falhas). O
    código foi restaurado depois de cada mutação.
  - Smoke com o modelo real (`pytest -m llm`), 4/4 passando.
- **Problemas que a validação pegou:**
  - Um release recusado por pré-condição entrava como "chamada já vista" e a tentativa
    válida seguinte encerrava como `NO_PROGRESS`.
  - Um teste que falhava antes de liberar uma thread presa travava a suíte; corrigido com
    uma fixture que sempre libera no teardown.
  - O `llama3.1:8b` respondeu "o envio foi liberado com sucesso" **sem chamar a tool**. O
    harness não aceitou (nenhuma liberação, `NO_PROGRESS`); agora o planner recebe uma
    correção ("texto não executa ações") e chama a tool.
- **Decisões tomadas pelo autor:** manter o LangGraph com `interrupt()` só nos testes; um
  único limite de 6 iterações; exigir acordo entre a intent do LLM e a regra antes de
  liberar; commits direto na `main`, sem PR, para caber no prazo.

## 9. Tempo dedicado

Implementação: **17:15 às 18:10, cerca de 55 minutos**, medido pelos commits (o primeiro
marca o início do relógio). O planejamento (documento de arquitetura, decisões e revisão do plano)
foi feito antes do início da contagem.

## 10. Como o harness controla a execução e limita iterações

| Limite | Padrão | Ao estourar |
|---|---|---|
| Iterações do planner (uma tool call cada) | 6 | `ABORTED` / `BUDGET_EXCEEDED` |
| Tempo total da execução | 90 s | `ABORTED` / `BUDGET_EXCEEDED` |
| Timeout por chamada ao LLM | 20 s | `ABORTED` / `PLANNER_ERROR` |
| Timeout por ferramenta | 2 s | Leitura: retry. Release: incerto, vai ao reconcile |
| Saída malformada do planner | 1 correção | `ABORTED` / `PLANNER_ERROR` |
| Mesma tool call proposta de novo | 1 vez | `ABORTED` / `NO_PROGRESS` |
| Planner encerra sem concluir | 1 correção (mesmo limite acima) | `ABORTED` / `NO_PROGRESS` |

- O budget e a detecção de repetição contam só o que o **planner** propõe. As chamadas
  internas do harness (1 retry por leitura, 1 leitura no reconcile) têm número fixo no código.
- Uma chamada recusada por pré-condição não conta como repetição: ela pode ficar válida
  depois das leituras. Um loop de recusas esbarra no limite de iterações.
- O `recursion_limit` do LangGraph fica como última rede; o budget encerra antes.

## 11. Separação entre regras críticas e decisões do LLM

| O LLM decide | O código decide |
|---|---|
| A intent (consulta, liberação, fora de escopo) | Se a intent concorda com a regra de palavras-chave |
| Qual tool chamar e com quais argumentos | Se os argumentos passam no schema (`ORD-\d{4}`) |
| Quando parar | Se o release pode acontecer: permissão, pedido `READY`, pagamento `CONFIRMED` |
| — | Retry, timeout, limites, classificação de erros |
| — | O texto da resposta (template por outcome) |

O policy gate (`policy.py`) é uma função pura que recebe o `RunContext`, as duas intents e os
dados **validados** de pedido e pagamento lidos nesta execução. Ele avalia as condições em
ordem e falha fechado, com um `reason` tipado. O release também só chega ao gate depois que
pedido e pagamento foram lidos com sucesso.

## 12. Cenários de avaliação

Todos verificam o que é observável de fora: as chamadas que o sistema externo recebeu, se o
release foi executado ou bloqueado, o outcome e o texto da resposta.

**Evals P0** (`tests/evals/test_p0.py`):

| # | Cenário | Verifica |
|---|---|---|
| 1 | Caminho feliz (planner roteirizado e por regras) | 3 chamadas em ordem, 1 único efeito, `RELEASED` |
| 2 | Pagamento pendente (roteirizado e por regras) | Só leituras, release nunca tentado, `BLOCKED_PAYMENT` |
| 3 | `get_payment` com timeout ou 5xx | 1 retry, `TOOL_FAILURE`, "Nenhuma liberação foi feita" |
| 4 | Planner tenta liberar com pagamento pendente | O release nunca chega ao sistema; recusa do gate no trace |
| 5 | Release recusado com 4xx | Uma tentativa só, `RELEASE_FAILED` |
| 6 | Texto só consulta, "LLM" tenta liberar | `INTENT_MISMATCH`, nada liberado |

**Release incerto** (`tests/evals/test_release_timeout.py`): falha depois do efeito confirmada
pelo reconcile; falha sem confirmação em `UNKNOWN` com o `run_id`; sucesso sem `release_id`
não conta; reconcile que também falha não afirma nada; nova solicitação após `UNKNOWN` não
libera duas vezes; timeout real com efeito atrasado (o reconcile vê `READY` e não conclui
falha); pausa e retomada com `interrupt()`, sem nova chamada externa.

**Testes de unidade** (`tests/unit/`): modelos, tools, intent, policy e harness (limites,
roteamento, correção do planner, timeout do LLM, tempo total).

**Smoke com LLM** (`tests/llm/`): caminho feliz, pagamento pendente, consulta e prompt
injection sem permissão.

## 13. Como executar e reproduzir os testes

```bash
uv run pytest                 # 113 testes: unidade + evals, sem LLM, ~1 s
uv run pytest tests/evals     # só os evals de comportamento
uv run pytest -m llm          # smoke com Ollama + llama3.1:8b (~80 s)
```

- A suíte padrão é determinística: planners sem LLM, sistema em memória e falhas injetadas
  como exceção. O timeout real é testado com um evento, sem `sleep`.
- Cada teste cria seu próprio sistema simulado (fixtures copiadas), então a ordem de execução
  não importa.
- O smoke depende do modelo e não é gate de regressão. Ele pula (skip) se o Ollama não estiver
  disponível. A métrica que importa nele é zero releases indevidos.

## 14. Timeout durante `release_shipment`

**O risco.** A chamada pode estourar o tempo depois que o sistema externo já liberou o envio.
Repetir às cegas pode liberar duas vezes. Reportar falha pode estar errado, e reportar sucesso
pode ser falso.

**O que a solução faz:**

1. **Toda chamada leva uma `idempotency_key`** derivada do pedido (`release:ORD-0001`). O
   sistema simulado devolve a confirmação original para uma chave repetida, sem novo efeito.
2. **Timeout, 5xx ou resposta ilegível não são falha nem sucesso.** O harness **não repete**
   o release.
3. **Reconcile:** uma leitura de `get_order`. Se o pedido está `SHIPPED`, o resultado é
   `RELEASED`, com a leitura como evidência.
4. **Sem confirmação, o resultado é `UNKNOWN`.** A resposta diz que o resultado é incerto,
   pede verificação manual e mostra o `run_id` para localizar a execução no log.
5. **Uma nova solicitação para o mesmo pedido não libera de novo:** o gate vê `SHIPPED` e
   bloqueia com `ALREADY_RELEASED`.

**Por que `READY` não prova que não liberou.** O sistema externo pode processar com atraso.
Um teste cobre isso: o timeout acontece, o reconcile ainda vê `READY` e o agente responde
`UNKNOWN`; só depois o efeito aparece.

**Por que não repetir, se o mock deduplica.** A deduplicação do mock existe para testar o caso
de duas chamadas com a mesma chave. Ela não representa uma garantia do sistema real. Retry
automático só é seguro se o sistema externo garantir deduplicação pela chave; com essa
garantia, o caminho preferido passa a ser repetir com a mesma chave e ler a resposta.

**Em produção:** idempotência garantida pelo sistema externo, um outbox com reconciliador
assíncrono para os `UNKNOWN`, e checkpointer persistente para retomar a execução com a decisão
de um operador. O node `escalate` já pausa com `interrupt()` e retoma com `released` ou
`not_released` (demonstrado nos testes com `MemorySaver`).

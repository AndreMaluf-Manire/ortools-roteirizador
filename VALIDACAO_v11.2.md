# VALIDAÇÃO v11.2 — Régua única de ETA (/recalculate-etas com OSRM)

**Data:** 2026-07-06 · **Branch:** `feat/recalc-etas-osrm-v11.2` · **Baseline:** v11.1 (`f35e887`)

## O que mudou (e SÓ isso)

O `/recalculate-etas` — endpoint chamado pelo front quando o usuário **edita uma rota
e clica em Salvar** — passa a usar a **mesma régua de tempo do `/optimize`**:

| | v11.1 (antes) | v11.2 (agora) |
|---|---|---|
| Tempo de viagem | Haversine ÷ velocidade do veículo (default 16 km/h) | **OSRM real de rua × fator trânsito 1.3** (igual `/optimize`) |
| Distância reportada | Haversine | **Estrada real (OSRM)** |
| Velocidade por veículo | aplicada | **ignorada sob OSRM** (igual `/optimize`) |
| Se OSRM cair/indisponível | — | **fallback automático = comportamento v11.1 exato** |

**NÃO mudou:** solver, `/optimize`, regra Vivenda, velocidades, custos, janelas,
qualquer regra de negócio. A mudança é 100% contida no endpoint de recálculo.

### Mecânica

- 1 chamada OSRM `/table` por rota editada (depot + paradas com coordenada).
- Timeout dedicado `OSRM_RECALC_TIMEOUT` (env, padrão **6s**) — o salvar do front
  aborta em 10s, então OSRM lento cede a vez pro Haversine em vez de travar o salvar.
- **Circuit breaker por lote:** na 1ª falha do OSRM, as demais rotas do request vão
  direto pro Haversine (não paga timeout N vezes).
- Campo novo `matrix_source` (`"osrm"` | `"haversine"`) por rota na resposta +
  contagem na `message`. Aditivo — edge function e front ignoram sem quebrar.
- Paradas sem coordenada: tolerância preservada (perna = 0, mantém última posição).

## ⚠️ EFEITO ESPERADO NO PRIMEIRO USO — NÃO É REGRESSÃO

**Os ETAs de rotas editadas vão SALTAR uma vez** na primeira edição pós-deploy:
a régua muda de Haversine 16 km/h (otimista no urbano, pessimista na estrada) pra
tempo real de rua. É o valor ficando **consistente com o que o `/optimize` mostra**
— antes, editar uma rota "mudava" os horários só pela troca de método de cálculo.
Se o operador estranhar horários diferentes após editar: **esperado, documentado aqui.**

## Latência (medida em 2026-07-06)

- OSRM prod, matriz 21×21 (rota de 20 paradas, pior caso realista): **~0,7s quente /
  ~1,8s fria** — medido do iMac pela internet; em prod é Railway→Railway (mesmo
  projeto), mais rápido. Muito abaixo do teto de 2-3s combinado.
- OSRM **morto** (DNS/host fora), lote de 2 rotas: **0,23s total** — falha rápida +
  circuit breaker. Pior caso teórico = 1× `OSRM_RECALC_TIMEOUT` (6s) se o host
  aceitar conexão e pendurar; ainda dentro dos 10s do front.

## Validação executada

### 1. Fallback = comportamento v11.1 exato ✅

Instância local com `OSRM_URL` inválida (timeout 2s), lote de 2 rotas × 4 paradas
(incluindo parada SEM coordenada e janela com espera). Algoritmo v11.1 reimplementado
de forma independente no teste: **8/8 paradas idênticas** (arrival, travel, distância),
`matrix_source="haversine"`, latência 0,23s.

### 2. Equivalência de régua /optimize × /recalculate-etas ✅

Rota realista na área de operação (CD Osasco + 8 clientes: Barueri, Cotia, Embu das
Artes, Taboão, Carapicuíba, Jandira, Itapevi, Santana de Parnaíba; 3 com janela),
motor local v11.2 + OSRM de produção:

- `/optimize` gerou rota de 8 paradas, 83,2 km, com espera de janela em Barueri.
- A MESMA sequência no `/recalculate-etas`: **8/8 paradas com chegada, tempo de
  viagem, espera e distância IDÊNTICOS** (ex.: Carapicuíba 313/313, Santana de
  Parnaíba 356/356, Barueri 393/393 com espera 27min dos dois lados, … Taboão
  574/574). `matrix_source="osrm"`.
- Ordem EDITADA (troca da 2ª com a 4ª parada, simulando a edição do usuário):
  ETAs recalculados na régua OSRM, valores mudam de forma coerente com o novo
  trajeto (ex.: última parada 574 → 610 pela ordem pior).

**Nota honesta sobre "rota real do banco":** o Supabase do roteirizador (`yxcwr…`)
segue **não autorizado no MCP** (limitação conhecida desde a Fase 2 da v11) e o ERP
não guarda coordenadas — não foi possível puxar uma rota real do banco desta máquina.
A validação usou municípios reais da área de operação com tempos OSRM reais de rua.
A prova de equivalência independe de QUAL rota é (mesma matriz dos dois lados =
mesmos números); se quiser o carimbo com dia real, basta editar qualquer rota após o
deploy e conferir `matrix_source: "osrm"` no log do Railway.

## Deploy / Smoke / Rollback

- **Deploy:** merge na main → Railway auto-deploya.
- **Smoke:** `GET /health` → `{"version": "11.2"}`; editar/salvar uma rota no board →
  log do Railway mostra `Rota <id>: régua osrm` e resposta `N via OSRM`.
- **Rollback:** `git revert <commit do merge> && git push origin main` → Railway
  redeploya v11.1. Alternativa cirúrgica sem deploy: **remover a env `OSRM_URL`**
  desliga a régua OSRM em TUDO (optimize + recalc) — só usar se quiser matar OSRM geral.

## RODADA 2 — EXECUTADA (2026-07-06, aprovação do André)

PR só de frontend (repo `roteirizadorfruleve`, branch `feat/routeeditor-regua-unica-osrm`).
**Zero toques no motor/Railway.**

1. ✅ **Régua 3 eliminada:** o botão "Atualizar ETA" do `RouteEditor.tsx` agora chama
   a edge `recalculate-etas` (a MESMA do board) em vez da `distance-matrix`/Google.
   Payload espelha o do `handleSaveChanges` do board. Junto:
   - Campo "Saída" do editor passa a inicializar do `routes.start_time` real do banco
     (antes assumia 06:00 fixo — os ETAs divergiam do board por construção);
   - Removido o limite de 20 paradas (era restrição dos 25 elementos da API Google;
     o motor não tem esse limite — board nunca teve);
   - Removidos `applyDurations` (cálculo local de ETA no front) e o cache de durações.
   - Badge adiantado/atrasado preservado (mapeado de `arrived_early`/`arrived_late`).
2. ✅ **Código morto removido:** `recalculateRouteETAs(routeId)` em
   `src/pages/Routes.tsx` (84 linhas, zero chamadores).
3. **Consumidores da edge `distance-matrix` verificados antes de remover a chamada:**
   grep no repo inteiro — o RouteEditor era o ÚNICO consumidor. A edge em si ficou
   no ar (sem chamadores); apagar a function é faxina futura opcional.

**Validação rodada 2:**
- `tsc --noEmit` limpo, `vite build` OK, testes do repo passam.
- Equivalência board × RouteEditor contra o MOTOR DE PRODUÇÃO: payload construído
  exatamente como cada tela constrói (mesma rota de 3 paradas, janela com espera de
  137min) → **respostas idênticas em todas as paradas** (chegada, viagem, espera,
  km), ambos `matrix_source: "osrm"`.
- ⏳ **Pendente pós-publish (Lovable, André):** abrir uma rota no RouteEditor,
  clicar "Atualizar ETA" e conferir que os horários batem com os do board para a
  mesma sequência. Front não tem auto-deploy do Git — precisa do publish.

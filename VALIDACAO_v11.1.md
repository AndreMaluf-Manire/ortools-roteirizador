# Validação v11.1 — guarda de regressão do primeiro dia real

**Deploy:** 2026-07-06 · commit `f35e887` (squash de `feat/consolidacao-veiculos-v11.1`)

## O que a v11.1 mudou (resumo)

Consolidação de veículos: custo fixo de carro real subiu de 500-10k para
100k-150k e o span foi calibrado junto (300/min). O solver agora só abre carro
novo quando o atual não comporta (capacidade/janela). fixed_driver virou
preferência suave (25k) — consolidação vence a preferência quando o carro do
freteiro não rodaria no dia. Limite de solução 150s (era 60s).

## Critérios da primeira roteirização real pós-deploy

| Critério | Esperado | Severidade |
|---|---|---|
| Entregas órfãs (unassigned) | **0 — absoluto** | Qualquer órfã = rollback imediato |
| Veículos usados | **≤ média recente da v11.0** no mesmo perfil de dia (referência: dias de ~80 entregas fechavam em ~6-8 carros; esperado 1-2 a menos) | Se usar MAIS carros que o normal, investigar antes do 2º disparo |
| Entregas Vivenda | 100% nos 2 FIORINOs do Marcelo Mota, balanceamento usual | Qualquer desvio = rollback |
| Não-Vivenda nos FIORINOs | Zero intruso | Qualquer intruso = rollback |
| Tempo de resposta do /optimize | ~150-160s (o solver GLS usa o limite inteiro de 150s por design; era ~60s na v11.0) | >200s ou timeout = investigar |
| Entregas fixed_driver | No carro do freteiro QUANDO ele roda no dia; podem cair em outro carro se consolidar dispensar o carro dele (comportamento novo, decidido em 2026-07-06) | Não é bug — é o desenho |

Baseline A/B local (ortools 9.8.3296, payload sintético 80 entregas / 446 cx /
10 veículos / 150s): v11.0 = 8 carros, v11.1 = 7 carros; 0 órfãs em ambos;
Vivenda 12/12 com distribuição idêntica; 0 intrusos.

## Rollback (1 comando, sem mexer no Railway)

```bash
git revert f35e887 && git push origin main
```

O Railway auto-deploya da main → produção volta ao comportamento v11.0
(custos antigos, fixed_driver rígido, 60s). Confirmar com:

```bash
curl -s https://ortools-roteirizador-production.up.railway.app/health
# deve voltar a responder "version": "11.0"
```

## Ajuste fino sem rollback

- Preferência de motorista mais "grudenta": subir `PREFERRED_DRIVER_PENALTY`
  (25k hoje; até ~90k continua sem forçar abertura de carro).
- Menos pressão contra rota longa: baixar `SPAN_COST_PER_MIN` (300 hoje; 100
  era o valor da v11.0).
- Resposta mais rápida em troca de consolidação: baixar `SOLUTION_TIME_LIMIT`.

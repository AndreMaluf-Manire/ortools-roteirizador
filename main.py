"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

VERSÃO 11.0 - PRECISÃO + ENCAIXA TODOS:
- Lê a time_matrix REAL enviada pelo roteirizador (Google Distance Matrix),
  com fallback Haversine por veículo quando não vier.
- Velocidade por veículo aplicada como multiplicador sobre a matriz real
  (ex.: veículo de estrada cobre mais rápido).
- Capacidade de caixas contada POR CLIENTE (0,5 + 0,5 do mesmo cliente = 1 cx;
  de clientes diferentes = 1 cx cada). Fim da inflação por arredondar cada linha.
- Janelas SOFT de verdade: respeita a janela, usa a margem só quando precisa,
  com penalidade suave de atraso (margem configurável via config).
- "Encaixa todos": veículos de reserva (carro adicional) com custo alto, usados
  só quando a frota real não fecha. DESLIGADO por padrão (config.allow_extra_vehicles).
- Hierarquia de custos por tipo de veículo (própria < agregado pequeno < grande).
- Vivenda configurável + exclusividade bidirecional (sem UUID hardcoded).
- Fixed Driver rígido sem ser sobrescrito pela regra Vivenda.
- /recalculate-etas PRESERVADO (recálculo só do que foi editado, sem solver).
- Autenticação opcional por segredo compartilhado (só ativa se ORTOOLS_API_KEY existir).
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Set
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import math
import time
import json
import urllib.request

app = FastAPI(
    title="OR-Tools Route Optimizer",
    description="API de otimização de rotas para o Roteirizador Manirê",
    version="11.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Autenticação opcional: SÓ é exigida quando ligada explicitamente via
# ORTOOLS_REQUIRE_AUTH=true. A mera presença de ORTOOLS_API_KEY (que já existe
# no Railway por legado) NÃO ativa a checagem — senão quebra o roteirizador,
# que hoje chama a API sem enviar token. Ativar só depois que o roteirizador
# passar a mandar o header Authorization.
ORTOOLS_API_KEY = os.getenv("ORTOOLS_API_KEY")
ORTOOLS_REQUIRE_AUTH = os.getenv("ORTOOLS_REQUIRE_AUTH", "false").lower() == "true"


def check_api_key(authorization: Optional[str]):
    if not (ORTOOLS_REQUIRE_AUTH and ORTOOLS_API_KEY):
        return  # API aberta (comportamento atual / padrão)
    if authorization != f"Bearer {ORTOOLS_API_KEY}":
        raise HTTPException(status_code=401, detail="Token inválido")


# Constantes
DEFAULT_SPEED_KMH = 16.0          # velocidade de referência (base do multiplicador por veículo)
MAX_TIME_HORIZON = 1440           # 24h em minutos
DEFAULT_SERVICE_TIME = 15
PENALTY_UNASSIGNED = 100_000_000_000_000   # 100 trilhões (último recurso: nunca deixar de fora)
SOLUTION_TIME_LIMIT = 60

# Custos fixos por tipo de veículo (quanto menor, mais "barato"/preferido)
COST_FROTA_PROPRIA = 500
COST_AGREGADO_PEQUENO = 5_000
COST_AGREGADO_GRANDE = 10_000
# Carro adicional/reserva: bem mais caro que qualquer real, mas MUITO mais barato que
# deixar cliente de fora -> o solver só adiciona um carro quando realmente precisa.
COST_EXTRA_VEHICLE = 1_000_000

# OSRM (tempos reais de carro pela rua). INERTE até OSRM_URL ser definida no ambiente.
OSRM_URL = os.getenv("OSRM_URL", "").strip()
OSRM_TRAFFIC_FACTOR = float(os.getenv("OSRM_TRAFFIC_FACTOR", "1.3"))
OSRM_TIMEOUT = float(os.getenv("OSRM_TIMEOUT", "25"))

# Padrão de saída do depot (minutos) para veículos SEM horário definido. 300 = 05:00.
GLOBAL_DEFAULT_DEPARTURE = 300

# Horário de saída por veículo (minutos). Prioridade: vehicle.start_time (cadastro) >
# este mapa temporário (por nome) > padrão global (GLOBAL_DEFAULT_DEPARTURE = 05:00).
# TEMPORÁRIO: até o roteirizador mandar start_time por veículo a partir do cadastro.
DEPARTURE_OVERRIDES = [
    ("IVECO", 240),   # Valmir — 04:00
    ("DUCATO", 180),  # Daniel — 03:00
    ("3/4", 180),     # Caminhão 3/4 (EDU) — 03:00
]


def vehicle_departure(vehicle, default_start: int) -> int:
    if vehicle.start_time is not None:
        return vehicle.start_time
    name = (vehicle.name or "").strip().upper()
    for key, minutes in DEPARTURE_OVERRIDES:
        if key in name:
            return minutes
    return default_start


# ============== MODELOS DE DADOS ==============

class Location(BaseModel):
    lat: float
    lng: float


class Depot(BaseModel):
    id: str
    name: str
    location: Location


class Customer(BaseModel):
    id: str
    name: str
    location: Location
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    service_time: int = DEFAULT_SERVICE_TIME


class Delivery(BaseModel):
    id: str
    customer_id: str
    customer_name: Optional[str] = None
    boxes: float = 0
    weight_kg: float = 0
    value: float = 0
    vehicle_id: Optional[str] = None
    preferred_driver_id: Optional[str] = None


class Vehicle(BaseModel):
    id: str
    name: str
    capacity_boxes: int
    capacity_kg: float = 99999
    average_speed_kmh: Optional[float] = None
    freteiro_id: Optional[str] = None
    # Opcional: classificação explícita de custo vinda do cadastro.
    # Valores: "propria" | "agregado_pequeno" | "agregado_grande".
    cost_tier: Optional[str] = None
    # Horário de saída do depot em minutos (ex.: 240 = 04:00). Se None, usa o global.
    start_time: Optional[int] = None


class RoutingRuleCondition(BaseModel):
    field: str
    operator: str
    value: str


class RoutingRuleAction(BaseModel):
    freteiro_id: Optional[str] = None
    vehicle_ids: Optional[List[str]] = None
    max_stops: Optional[int] = None
    distribution_mode: Optional[str] = None


class RoutingRule(BaseModel):
    type: str
    priority: int = 0
    condition: Optional[RoutingRuleCondition] = None
    action: Optional[RoutingRuleAction] = None
    vehicle_ids: Optional[List[str]] = None


class VivendaruleConfig(BaseModel):
    enabled: bool = False
    vehicle_ids: List[str] = []
    keyword: str = "vivenda"
    balance_mode: str = "volume"


class OptimizationConfig(BaseModel):
    soft_time_window_tolerance: int = 15          # margem (min) que a janela pode estourar
    late_penalty_per_min: int = 50                # penalidade suave por minuto de atraso
    vivenda_rule: Optional[VivendaruleConfig] = None
    force_allocation: bool = True
    # "Encaixa todos" com carro adicional — DESLIGADO por padrão (segurança).
    allow_extra_vehicles: bool = False
    extra_vehicles_count: int = 5                 # quantos carros de reserva podem entrar
    extra_vehicle_capacity_boxes: int = 0         # 0 = auto (= maior capacidade da frota real)


class OptimizeRequest(BaseModel):
    depot: Depot
    customers: List[Customer]
    deliveries: List[Delivery]
    vehicles: List[Vehicle]
    start_time: int = 360
    mode: str = "minimize_vehicles"
    delivery_groups: Optional[List[List[str]]] = None
    routing_rules: Optional[List[RoutingRule]] = None
    config: Optional[OptimizationConfig] = None
    # Matriz de tempo (minutos) já calculada pelo roteirizador (Google ou fallback).
    # Ordem: [depot, ...customers] na MESMA ordem de `customers`.
    time_matrix: Optional[List[List[int]]] = None
    max_route_duration: Optional[int] = None      # informativo por enquanto


class RouteStop(BaseModel):
    delivery_id: str
    customer_id: str
    customer_name: str
    location: Location
    sequence: int
    arrival_time: int
    departure_time: int
    service_time: int
    window_start: int
    window_end: int
    travel_time: int
    distance_km: float
    wait_time: int
    arrived_early: bool = False
    arrived_late: bool = False
    boxes: float = 0
    weight_kg: float = 0
    value: float = 0


class Route(BaseModel):
    vehicle_id: str
    vehicle_name: str
    stops: List[RouteStop]
    total_boxes: float
    total_weight: float
    total_value: float
    total_distance_km: float
    total_time_minutes: int
    start_time: int
    end_time: int
    is_extra_vehicle: bool = False   # True = carro adicional sugerido (não é da frota real)


class OptimizeResponse(BaseModel):
    success: bool
    message: str
    routes: List[Route]
    unassigned_deliveries: List[str]
    vehicles_used: int
    total_deliveries: int
    total_value: float
    optimization_time_ms: int
    extra_vehicles_used: int = 0     # quantos carros adicionais foram necessários


# ============== FUNÇÕES AUXILIARES ==============

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def minutes_to_time(minutes: int) -> str:
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def create_time_matrix(locations: List[Location], speed_kmh: float) -> List[List[int]]:
    n = len(locations)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = haversine_distance(
                    locations[i].lat, locations[i].lng,
                    locations[j].lat, locations[j].lng
                )
                matrix[i][j] = int(math.ceil((dist / speed_kmh) * 60))
    return matrix


def scale_matrix(base: List[List[int]], factor: float) -> List[List[int]]:
    """Escala uma matriz de tempos por um fator (para velocidade por veículo)."""
    if abs(factor - 1.0) < 1e-9:
        return [row[:] for row in base]
    n = len(base)
    return [[int(math.ceil(base[i][j] * factor)) for j in range(n)] for i in range(n)]


def fetch_osrm_matrix(locations: List[Location]):
    """
    Via OSRM, retorna (tempos_min, distancias_km) REAIS de carro pela rua:
      - tempos em minutos, já com fator de trânsito;
      - distâncias em km (estrada real), sem fator (distância não muda com trânsito).
    Retorna None se OSRM_URL não estiver setada ou falhar (cai no fallback). INERTE por padrão.
    """
    if not OSRM_URL:
        return None
    try:
        coords = ";".join(f"{loc.lng},{loc.lat}" for loc in locations)  # OSRM usa lng,lat
        url = f"{OSRM_URL.rstrip('/')}/table/v1/driving/{coords}?annotations=duration,distance"
        with urllib.request.urlopen(url, timeout=OSRM_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("code") != "Ok" or "durations" not in data:
            print(f"  OSRM resposta inesperada: {data.get('code')}")
            return None
        dur = data["durations"]
        n = len(locations)
        if len(dur) != n:
            return None
        time_min = [
            [int(math.ceil(((dur[i][j] or 0) / 60.0) * OSRM_TRAFFIC_FACTOR)) for j in range(n)]
            for i in range(n)
        ]
        dist_raw = data.get("distances")
        dist_km = None
        if dist_raw and len(dist_raw) == n:
            dist_km = [[round((dist_raw[i][j] or 0) / 1000.0, 2) for j in range(n)] for i in range(n)]
        return (time_min, dist_km)
    except Exception as e:
        print(f"  OSRM indisponível ({e}) — usando fallback")
        return None


def build_time_matrices(
    locations: List[Location],
    vehicles: List["Vehicle"],
    provided_matrix: Optional[List[List[int]]],
    default_speed: float,
    apply_vehicle_speed: bool = True,
) -> (Dict[int, List[List[int]]], List[List[int]], bool):
    """
    Retorna (matrizes_por_veiculo, matriz_base, usou_matriz_real).
    - Se veio time_matrix do roteirizador e bate o tamanho: usa ela como base (tempo real)
      e aplica velocidade por veículo como multiplicador (referência = DEFAULT_SPEED_KMH).
    - Senão: cai no Haversine por veículo (comportamento antigo).
    """
    n = len(locations)
    used_real = False
    base_matrix: List[List[int]]

    if provided_matrix and len(provided_matrix) == n and all(len(r) == n for r in provided_matrix):
        base_matrix = [[int(provided_matrix[i][j]) for j in range(n)] for i in range(n)]
        used_real = True
    else:
        base_matrix = create_time_matrix(locations, default_speed)

    matrices: Dict[int, List[List[int]]] = {}
    for idx, vehicle in enumerate(vehicles):
        speed = vehicle.average_speed_kmh if vehicle.average_speed_kmh else default_speed
        if used_real and not apply_vehicle_speed:
            # OSRM já dá o tempo real de carro -> usa igual pra todos (sem multiplicador de velocidade).
            matrices[idx] = [row[:] for row in base_matrix]
        elif used_real:
            # Matriz real é "carro normal" ~ default_speed. Veículo mais rápido -> tempos menores.
            factor = default_speed / speed if speed else 1.0
            matrices[idx] = scale_matrix(base_matrix, factor)
        else:
            matrices[idx] = create_time_matrix(locations, speed)
    return matrices, base_matrix, used_real


def create_distance_matrix(locations: List[Location]) -> List[List[float]]:
    n = len(locations)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_distance(
                    locations[i].lat, locations[i].lng,
                    locations[j].lat, locations[j].lng
                )
    return matrix


def classify_vehicle_cost(vehicle: Vehicle) -> int:
    """
    Custo fixo do veículo. Usa cost_tier explícito se vier do cadastro;
    senão cai numa heurística por nome (menos confiável).
    """
    if vehicle.cost_tier:
        tier = vehicle.cost_tier.strip().lower()
        if tier == "propria":
            return COST_FROTA_PROPRIA
        if tier == "agregado_pequeno":
            return COST_AGREGADO_PEQUENO
        if tier == "agregado_grande":
            return COST_AGREGADO_GRANDE

    vehicle_name = (vehicle.name or "").upper()
    vehicle_id = (vehicle.id or "").upper()

    agregado_keywords = ['OUTROS', 'FRETEIRO', 'AGREGADO', 'TERCEIRO']
    is_agregado = any(kw in vehicle_name or kw in vehicle_id for kw in agregado_keywords)
    if not is_agregado:
        return COST_FROTA_PROPRIA

    agregado_grande_keywords = ['HR', 'IVECO', 'CAMINHÃO', 'CAMINHAO', 'VUC', 'TRUCK', '3/4', 'TOCO']
    is_grande = any(kw in vehicle_name for kw in agregado_grande_keywords)
    return COST_AGREGADO_GRANDE if is_grande else COST_AGREGADO_PEQUENO


def apply_vivenda_rule(
    deliveries: List[Delivery],
    config: VivendaruleConfig,
):
    """Distribui entregas Vivenda nos veículos exclusivos (bin packing por volume/peso)."""
    vivenda_deliveries = []
    non_vivenda_deliveries = []
    keyword = config.keyword.lower()

    for delivery in deliveries:
        customer_name = (delivery.customer_name or "").lower()
        if keyword in customer_name:
            vivenda_deliveries.append(delivery)
        else:
            non_vivenda_deliveries.append(delivery)

    print(f"\n=== REGRA VIVENDA (EXCLUSIVIDADE BIDIRECIONAL) ===")
    print(f"  Keyword: '{config.keyword}' | Veículos: {config.vehicle_ids}")
    print(f"  Vivenda: {len(vivenda_deliveries)} | Não-Vivenda: {len(non_vivenda_deliveries)}")

    if len(vivenda_deliveries) == 0:
        return [], deliveries, set(config.vehicle_ids)
    if len(config.vehicle_ids) == 0:
        print("  ❌ Regra ativa sem veículos definidos!")
        return vivenda_deliveries, non_vivenda_deliveries, set()

    key_fn = (lambda d: d.weight_kg) if config.balance_mode == "weight" else (lambda d: d.boxes)
    vivenda_deliveries.sort(key=key_fn, reverse=True)
    vehicle_loads = {vid: 0.0 for vid in config.vehicle_ids}
    for delivery in vivenda_deliveries:
        lightest = min(vehicle_loads, key=vehicle_loads.get)
        delivery.vehicle_id = lightest
        vehicle_loads[lightest] += key_fn(delivery)
    print(f"  Balanceamento ({config.balance_mode}): {[round(l, 1) for l in vehicle_loads.values()]}")

    return vivenda_deliveries, non_vivenda_deliveries, set(config.vehicle_ids)


# ============== OTIMIZAÇÃO ==============

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_routes(request: OptimizeRequest, authorization: Optional[str] = Header(None)):
    check_api_key(authorization)
    start_time_ms = time.time() * 1000

    print(f"\n{'='*60}")
    print(f"=== OTIMIZAÇÃO v11.0 - PRECISÃO + ENCAIXA TODOS ===")
    print(f"{'='*60}")
    print(f"Depot: {request.depot.name} | Entregas: {len(request.deliveries)} | Veículos: {len(request.vehicles)}")
    print(f"Modo: {request.mode} | Penalidade não-atendimento: {PENALTY_UNASSIGNED:,}")

    if len(request.deliveries) == 0:
        return OptimizeResponse(
            success=True, message="Nenhuma entrega para otimizar", routes=[],
            unassigned_deliveries=[], vehicles_used=0, total_deliveries=0,
            total_value=0, optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )

    cfg = request.config or OptimizationConfig()
    margin = max(0, cfg.soft_time_window_tolerance)
    print(f"Margem de janela: {margin}min | Atraso penalizado: {cfg.late_penalty_per_min}/min")
    print(f"Carro adicional: {'LIGADO' if cfg.allow_extra_vehicles else 'desligado'}")

    default_speed = DEFAULT_SPEED_KMH
    customer_map = {c.id: c for c in request.customers}

    # ===== REGRA VIVENDA =====
    vivenda_deliveries = []
    non_vivenda_deliveries = request.deliveries
    vivenda_vehicle_set: Set[str] = set()
    vivenda_keyword = ""
    if cfg.vivenda_rule and cfg.vivenda_rule.enabled:
        vivenda_keyword = cfg.vivenda_rule.keyword.lower()
        vivenda_deliveries, non_vivenda_deliveries, vivenda_vehicle_set = apply_vivenda_rule(
            request.deliveries, cfg.vivenda_rule
        )
    else:
        print("\n=== REGRA VIVENDA: DESATIVADA ===")

    # ===== LOCALIZAÇÕES =====
    locations = [request.depot.location]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if customer:
            locations.append(customer.location)
        else:
            # Cliente ausente no cadastro: avisa e usa depot (não silencioso).
            print(f"  ⚠️ Cliente não encontrado para entrega {delivery.id} (customer_id={delivery.customer_id}) — usando depot")
            locations.append(Location(lat=request.depot.location.lat, lng=request.depot.location.lng))
    num_locations = len(locations)

    # ===== JANELA INFINITA (Vivenda + Fixed Driver) =====
    infinite_window_ids: Set[str] = set()
    if vivenda_keyword:
        for delivery in request.deliveries:
            if vivenda_keyword in (delivery.customer_name or "").lower():
                infinite_window_ids.add(delivery.id)

    # ===== REGRAS DE ROTEAMENTO =====
    fixed_assignments: Dict[str, str] = {}   # delivery_id -> vehicle_id (rígido)
    if request.routing_rules:
        print(f"\n=== REGRAS DE ROTEAMENTO ({len(request.routing_rules)}) ===")
        for rule in sorted(request.routing_rules, key=lambda r: r.priority, reverse=True):
            if rule.type == "fixed_driver":
                if not rule.action or not rule.action.freteiro_id or not rule.condition:
                    continue
                freteiro_id = rule.action.freteiro_id
                target_vehicle_ids = rule.action.vehicle_ids or None
                for delivery in request.deliveries:
                    if delivery.vehicle_id:
                        continue
                    cname = (delivery.customer_name or "").lower()
                    val = rule.condition.value.lower()
                    op = rule.condition.operator
                    matches = (
                        (op == "contains" and val in cname) or
                        (op == "starts_with" and cname.startswith(val)) or
                        (op == "equals" and cname == val)
                    )
                    if matches:
                        target_vehicle = next(
                            (v for v in request.vehicles
                             if v.freteiro_id == freteiro_id and (target_vehicle_ids is None or v.id in target_vehicle_ids)),
                            None
                        )
                        if target_vehicle:
                            fixed_assignments[delivery.id] = target_vehicle.id
                            infinite_window_ids.add(delivery.id)

            elif rule.type == "group_by_name":
                if not rule.condition:
                    continue
                group = []
                for delivery in request.deliveries:
                    cname = (delivery.customer_name or "").lower()
                    val = rule.condition.value.lower()
                    op = rule.condition.operator
                    if (op == "contains" and val in cname) or \
                       (op == "starts_with" and cname.startswith(val)) or \
                       (op == "equals" and cname == val):
                        group.append(delivery.id)
                if len(group) > 1:
                    request.delivery_groups = (request.delivery_groups or []) + [group]
        print(f"  Fixed Driver: {len(fixed_assignments)} entrega(s) travada(s)")

    # ===== JANELAS DE TEMPO =====
    time_windows = [(0, MAX_TIME_HORIZON)]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        if delivery.id in infinite_window_ids:
            time_windows.append((0, MAX_TIME_HORIZON))
        elif customer and customer.window_start is not None and customer.window_end is not None:
            time_windows.append((customer.window_start, customer.window_end))
        else:
            time_windows.append((0, MAX_TIME_HORIZON))

    service_times = [0]
    for delivery in request.deliveries:
        customer = customer_map.get(delivery.customer_id)
        service_times.append(customer.service_time if customer else DEFAULT_SERVICE_TIME)

    # ===== VEÍCULOS (reais + reserva opcional) =====
    if len(request.vehicles) == 0:
        return OptimizeResponse(
            success=False, message="Nenhum veículo disponível", routes=[],
            unassigned_deliveries=[d.id for d in request.deliveries], vehicles_used=0,
            total_deliveries=len(request.deliveries), total_value=0,
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )

    vehicles: List[Vehicle] = list(request.vehicles)
    num_real_vehicles = len(vehicles)

    extra_vehicle_ids: Set[str] = set()
    if cfg.allow_extra_vehicles and cfg.extra_vehicles_count > 0:
        extra_cap = cfg.extra_vehicle_capacity_boxes or max((v.capacity_boxes for v in request.vehicles), default=999)
        for i in range(cfg.extra_vehicles_count):
            vid = f"EXTRA-{i+1}"
            extra_vehicle_ids.add(vid)
            vehicles.append(Vehicle(
                id=vid, name=f"🚐 CARRO ADICIONAL {i+1}",
                capacity_boxes=int(extra_cap), capacity_kg=99999,
                average_speed_kmh=default_speed, cost_tier="agregado_grande"
            ))
        print(f"\n=== CARROS DE RESERVA: {cfg.extra_vehicles_count} (cap {extra_cap} cx, custo {COST_EXTRA_VEHICLE:,}) ===")

    num_vehicles = len(vehicles)

    # Custos por veículo (real = hierarquia; extra = custo alto)
    vehicle_costs = []
    for v in vehicles:
        vehicle_costs.append(COST_EXTRA_VEHICLE if v.id in extra_vehicle_ids else classify_vehicle_cost(v))

    # ===== MATRIZES =====
    # Prioridade: OSRM (rua real, sem limite de 25) > matriz do roteirizador > Haversine.
    osrm = fetch_osrm_matrix(locations)
    osrm_time = osrm[0] if osrm else None
    osrm_dist = osrm[1] if osrm else None
    base_provided = osrm_time if osrm_time else request.time_matrix
    if osrm_time:
        matrix_source = f"OSRM real (fator trânsito {OSRM_TRAFFIC_FACTOR})"
    elif request.time_matrix:
        matrix_source = "roteirizador (payload)"
    else:
        matrix_source = "Haversine (fallback)"
    time_matrices, base_time_matrix, used_real = build_time_matrices(
        locations, vehicles, base_provided, default_speed,
        apply_vehicle_speed=(osrm_time is None),  # sob OSRM, ignora velocidade por veículo
    )
    print(f"Matriz de tempo: {matrix_source}")
    # Distância reportada: estrada real do OSRM quando disponível; senão Haversine.
    distance_matrix = osrm_dist if osrm_dist else create_distance_matrix(locations)

    delivery_id_to_node = {d.id: i + 1 for i, d in enumerate(request.deliveries)}
    vehicle_id_to_idx = {v.id: i for i, v in enumerate(vehicles)}

    # ===== DEMANDA (CAIXAS POR CLIENTE) =====
    # 0,5 + 0,5 do MESMO cliente = 1; de clientes diferentes = 1 cada.
    customer_total_boxes: Dict[str, float] = {}
    for d in request.deliveries:
        customer_total_boxes[d.customer_id] = customer_total_boxes.get(d.customer_id, 0.0) + (d.boxes or 0)
    node_demand = [0] * num_locations
    seen_customers: Set[str] = set()
    for idx, d in enumerate(request.deliveries):
        node = idx + 1
        if d.customer_id not in seen_customers:
            seen_customers.add(d.customer_id)
            node_demand[node] = int(math.ceil(customer_total_boxes[d.customer_id]))
        else:
            node_demand[node] = 0  # caixas já contadas na 1ª parada do cliente

    # ===== MODELO OR-TOOLS =====
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def make_time_callback(vehicle_idx):
        tm = time_matrices.get(vehicle_idx, base_time_matrix)
        def cb(from_index, to_index):
            fn = manager.IndexToNode(from_index)
            tn = manager.IndexToNode(to_index)
            return tm[fn][tn] + (service_times[tn] if tn > 0 else 0)
        return cb

    time_callback_indices = []
    for vehicle_idx in range(num_vehicles):
        time_callback_indices.append(routing.RegisterTransitCallback(make_time_callback(vehicle_idx)))

    def base_time_callback(from_index, to_index):
        fn = manager.IndexToNode(from_index)
        tn = manager.IndexToNode(to_index)
        return base_time_matrix[fn][tn] + (service_times[tn] if tn > 0 else 0)

    base_time_callback_index = routing.RegisterTransitCallback(base_time_callback)

    routing.AddDimension(base_time_callback_index, MAX_TIME_HORIZON, MAX_TIME_HORIZON, False, "Time")
    time_dimension = routing.GetDimensionOrDie("Time")

    # Janelas: range rígido [ws-margem, we+margem]; penalidade suave após we (atraso só se precisar).
    for node in range(num_locations):
        index = manager.NodeToIndex(node)
        ws, we = time_windows[node]
        if we < MAX_TIME_HORIZON:
            time_dimension.CumulVar(index).SetRange(max(0, ws - margin), min(MAX_TIME_HORIZON, we + margin))
            if cfg.late_penalty_per_min > 0:
                time_dimension.SetCumulVarSoftUpperBound(index, we, cfg.late_penalty_per_min)
        else:
            time_dimension.CumulVar(index).SetRange(ws, we)

    # Horário de saída por veículo (Valmir 04h, Daniel/EDU 03h, demais 06h — via mapa/cadastro)
    vehicle_starts = [vehicle_departure(v, GLOBAL_DEFAULT_DEPARTURE) for v in vehicles]
    for vehicle_idx in range(num_vehicles):
        time_dimension.SetSpanCostCoefficientForVehicle(100, vehicle_idx)
        start_index = routing.Start(vehicle_idx)
        st = vehicle_starts[vehicle_idx]
        time_dimension.CumulVar(start_index).SetRange(st, st)
        if st != request.start_time:
            print(f"  Saída especial: {vehicles[vehicle_idx].name} -> {minutes_to_time(st)}")

    # Capacidade (caixas por cliente)
    def demand_callback(from_index):
        return node_demand[manager.IndexToNode(from_index)]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [v.capacity_boxes for v in vehicles], True, "Capacity"
    )

    # Custos por veículo (arco = tempo; fixo = hierarquia/reserva)
    for vehicle_idx in range(num_vehicles):
        routing.SetArcCostEvaluatorOfVehicle(time_callback_indices[vehicle_idx], vehicle_idx)
        routing.SetFixedCostOfVehicle(vehicle_costs[vehicle_idx], vehicle_idx)

    # ===== VIVENDA: exclusividade bidirecional =====
    if vivenda_deliveries and vivenda_vehicle_set:
        vivenda_idx = [vehicle_id_to_idx[v] for v in vivenda_vehicle_set if v in vehicle_id_to_idx]
        if not vivenda_idx:
            return OptimizeResponse(
                success=False, message="Veículos da regra Vivenda não existem na frota configurada",
                routes=[], unassigned_deliveries=[d.id for d in request.deliveries], vehicles_used=0,
                total_deliveries=len(request.deliveries), total_value=0,
                optimization_time_ms=int(time.time() * 1000 - start_time_ms)
            )
        for delivery in vivenda_deliveries:
            if delivery.vehicle_id and delivery.vehicle_id in vehicle_id_to_idx:
                index = manager.NodeToIndex(delivery_id_to_node[delivery.id])
                routing.SetAllowedVehiclesForIndex([vehicle_id_to_idx[delivery.vehicle_id]], index)
        # Não-Vivenda: proibir nos veículos exclusivos — exceto fixed drivers (não sobrescrever).
        non_vivenda_idx = [i for i in range(num_vehicles) if i not in vivenda_idx]
        if non_vivenda_idx:
            for delivery in non_vivenda_deliveries:
                if delivery.id in fixed_assignments:
                    continue
                index = manager.NodeToIndex(delivery_id_to_node[delivery.id])
                routing.SetAllowedVehiclesForIndex(non_vivenda_idx, index)
        else:
            print("  ⚠️ TODOS os veículos são Vivenda — não-Vivenda podem ficar órfãs")

    # ===== FIXED DRIVER (rígido) — aplicado por último para não ser sobrescrito =====
    if fixed_assignments:
        for delivery_id, vehicle_id in fixed_assignments.items():
            if delivery_id in delivery_id_to_node and vehicle_id in vehicle_id_to_idx:
                index = manager.NodeToIndex(delivery_id_to_node[delivery_id])
                routing.SetAllowedVehiclesForIndex([vehicle_id_to_idx[vehicle_id]], index)

    # ===== GRUPOS DE ENTREGAS (mesmo veículo) =====
    if request.delivery_groups:
        for group in request.delivery_groups:
            nodes = [delivery_id_to_node[d] for d in group if d in delivery_id_to_node]
            if len(nodes) >= 2:
                first = manager.NodeToIndex(nodes[0])
                for n in nodes[1:]:
                    routing.solver().Add(routing.VehicleVar(first) == routing.VehicleVar(manager.NodeToIndex(n)))

    # ===== PENALIDADE: nunca deixar de fora (último recurso) =====
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], PENALTY_UNASSIGNED)

    # ===== RESOLVER =====
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(SOLUTION_TIME_LIMIT)

    print(f"\n=== RESOLVENDO (limite: {SOLUTION_TIME_LIMIT}s) ===")
    solution = routing.SolveWithParameters(search_parameters)

    if not solution:
        print("ERRO: Não encontrou solução!")
        return OptimizeResponse(
            success=False,
            message="Não foi possível encontrar uma solução. Verifique janelas de tempo e capacidades.",
            routes=[], unassigned_deliveries=[d.id for d in request.deliveries], vehicles_used=0,
            total_deliveries=len(request.deliveries), total_value=0,
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )

    print("✅ Solução encontrada!")

    # ===== EXTRAIR SOLUÇÃO =====
    routes = []
    vehicles_used = 0
    extra_vehicles_used = 0
    total_value_all = 0.0
    all_assigned = set()

    for vehicle_idx in range(num_vehicles):
        index = routing.Start(vehicle_idx)
        route_nodes = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node > 0:
                route_nodes.append(node)
            index = solution.Value(routing.NextVar(index))
        if not route_nodes:
            continue

        vehicle = vehicles[vehicle_idx]
        is_extra = vehicle.id in extra_vehicle_ids
        tm = time_matrices.get(vehicle_idx, base_time_matrix)

        stops = []
        route_boxes = route_weight = route_value = route_distance = 0.0
        veh_start = vehicle_starts[vehicle_idx]
        current_time = veh_start
        prev_node = 0

        for seq, node in enumerate(route_nodes):
            delivery = request.deliveries[node - 1]
            customer = customer_map.get(delivery.customer_id)
            all_assigned.add(delivery.id)

            travel_time = tm[prev_node][node]
            distance = distance_matrix[prev_node][node]
            arrival_time = current_time + travel_time
            ws, we = time_windows[node]

            wait_time = 0
            arrived_early = arrived_late = False
            if arrival_time < ws:
                wait_time = ws - arrival_time
                arrived_early = True
                service_start = ws
            else:
                service_start = arrival_time
                if arrival_time > we:
                    arrived_late = True

            stop_service = service_times[node]
            departure_time = service_start + stop_service

            route_boxes += delivery.boxes
            route_weight += delivery.weight_kg
            route_value += delivery.value
            route_distance += distance

            stops.append(RouteStop(
                delivery_id=delivery.id, customer_id=delivery.customer_id,
                customer_name=customer.name if customer else "?",
                location=locations[node], sequence=seq + 1,
                arrival_time=arrival_time, departure_time=departure_time,
                service_time=stop_service, window_start=ws, window_end=we,
                travel_time=travel_time, distance_km=round(distance, 1),
                wait_time=wait_time, arrived_early=arrived_early, arrived_late=arrived_late,
                boxes=delivery.boxes, weight_kg=delivery.weight_kg, value=delivery.value
            ))
            current_time = departure_time
            prev_node = node

        routes.append(Route(
            vehicle_id=vehicle.id, vehicle_name=vehicle.name, stops=stops,
            total_boxes=round(route_boxes, 1), total_weight=round(route_weight, 1),
            total_value=round(route_value, 2), total_distance_km=round(route_distance, 1),
            total_time_minutes=current_time - veh_start,
            start_time=veh_start, end_time=current_time,
            is_extra_vehicle=is_extra
        ))
        vehicles_used += 1
        if is_extra:
            extra_vehicles_used += 1
        total_value_all += route_value
        tag = " 🚐 ADICIONAL" if is_extra else ""
        print(f"  ROTA {vehicle.name}{tag}: {len(stops)} entregas, {round(route_boxes,1)} cx, {round(route_distance,1)} km")

    unassigned = []
    for delivery in request.deliveries:
        if delivery.id not in all_assigned:
            unassigned.append(delivery.id)
            customer = customer_map.get(delivery.customer_id)
            print(f"⚠️ NÃO ALOCADA: {customer.name if customer else '?'} (ID: {delivery.id})")

    optimization_time = int(time.time() * 1000 - start_time_ms)

    print(f"\n=== RESUMO FINAL ===")
    print(f"Veículos: {vehicles_used} ({vehicles_used - extra_vehicles_used} reais + {extra_vehicles_used} adicionais)")
    print(f"Alocadas: {len(all_assigned)}/{len(request.deliveries)} | Pendentes: {len(unassigned)} | {optimization_time}ms")

    msg = f"Otimização concluída. {vehicles_used} veículos ({extra_vehicles_used} adicional(is)), {len(all_assigned)} entregas, {len(unassigned)} pendentes."
    return OptimizeResponse(
        success=True, message=msg, routes=routes, unassigned_deliveries=unassigned,
        vehicles_used=vehicles_used, total_deliveries=len(request.deliveries),
        total_value=round(total_value_all, 2), optimization_time_ms=optimization_time,
        extra_vehicles_used=extra_vehicles_used
    )


# ============== MODELOS RECALCULATE-ETAS ==============

class RecalcStop(BaseModel):
    """Parada na ordem definida pelo usuário"""
    delivery_id: str
    customer_name: Optional[str] = None
    lat: Optional[float] = None      # tolerante: cliente sem coordenada não derruba o lote
    lng: Optional[float] = None
    service_time: int = DEFAULT_SERVICE_TIME
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    boxes: float = 0
    weight_kg: float = 0
    value: float = 0


class RecalcRoute(BaseModel):
    """Rota editada para recálculo de ETAs"""
    route_id: str
    vehicle_id: Optional[str] = None   # tolerante: rota "sem veículo" / carro adicional não derruba o lote
    vehicle_speed_kmh: Optional[float] = None
    start_time: int = 360
    depot_lat: float
    depot_lng: float
    stops: List[RecalcStop]


class RecalcRequest(BaseModel):
    """Request: apenas as rotas que foram editadas"""
    routes: List[RecalcRoute]


class RecalcStopResult(BaseModel):
    delivery_id: str
    sequence: int
    arrival_time: int
    departure_time: int
    travel_time: int
    wait_time: int
    distance_km: float
    arrived_early: bool = False
    arrived_late: bool = False


class RecalcRouteResult(BaseModel):
    route_id: str
    vehicle_id: Optional[str] = None
    stops: List[RecalcStopResult]
    total_distance_km: float
    total_time_minutes: int
    start_time: int
    end_time: int


class RecalcResponse(BaseModel):
    success: bool
    message: str
    routes: List[RecalcRouteResult]
    recalc_time_ms: int


# ============== ENDPOINT RECALCULATE-ETAS ==============
# PRESERVADO da versão estável: recalcula ETAs sequencialmente para rotas editadas
# manualmente, SEM solver. Enviar APENAS as rotas que sofreram edição.

@app.post("/recalculate-etas", response_model=RecalcResponse)
async def recalculate_etas(request: RecalcRequest, authorization: Optional[str] = Header(None)):
    check_api_key(authorization)
    start_ms = time.time() * 1000

    if not request.routes:
        return RecalcResponse(success=True, message="Nenhuma rota para recalcular", routes=[], recalc_time_ms=0)

    print(f"\n=== RECALCULATE-ETAS ({len(request.routes)} rota(s)) ===")
    result_routes: List[RecalcRouteResult] = []

    for route in request.routes:
        speed = route.vehicle_speed_kmh or DEFAULT_SPEED_KMH
        current_time = route.start_time
        prev_lat, prev_lng = route.depot_lat, route.depot_lng
        stop_results: List[RecalcStopResult] = []
        total_distance = 0.0

        for seq, stop in enumerate(route.stops):
            # Parada (ou ponto anterior) sem coordenada: não dá pra medir deslocamento —
            # assume 0 e mantém a última posição conhecida, sem derrubar o lote inteiro.
            if stop.lat is None or stop.lng is None or prev_lat is None or prev_lng is None:
                dist = 0.0
            else:
                dist = haversine_distance(prev_lat, prev_lng, stop.lat, stop.lng)
            travel_time = int(math.ceil((dist / speed) * 60))
            arrival_time = current_time + travel_time

            wait_time = 0
            arrived_early = arrived_late = False
            if stop.window_start is not None and arrival_time < stop.window_start:
                wait_time = stop.window_start - arrival_time
                arrived_early = True
                service_start = stop.window_start
            else:
                service_start = arrival_time
                if stop.window_end is not None and arrival_time > stop.window_end:
                    arrived_late = True

            departure_time = service_start + stop.service_time
            total_distance += dist

            stop_results.append(RecalcStopResult(
                delivery_id=stop.delivery_id, sequence=seq + 1,
                arrival_time=arrival_time, departure_time=departure_time,
                travel_time=travel_time, wait_time=wait_time,
                distance_km=round(dist, 1), arrived_early=arrived_early, arrived_late=arrived_late
            ))
            current_time = departure_time
            if stop.lat is not None and stop.lng is not None:
                prev_lat, prev_lng = stop.lat, stop.lng

        result_routes.append(RecalcRouteResult(
            route_id=route.route_id, vehicle_id=route.vehicle_id, stops=stop_results,
            total_distance_km=round(total_distance, 1),
            total_time_minutes=current_time - route.start_time,
            start_time=route.start_time, end_time=current_time
        ))

    recalc_time = int(time.time() * 1000 - start_ms)
    return RecalcResponse(
        success=True, message=f"{len(result_routes)} rota(s) recalculada(s)",
        routes=result_routes, recalc_time_ms=recalc_time
    )


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    return {"status": "ok", "service": "OR-Tools Route Optimizer", "version": "11.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "version": "11.0"}


# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

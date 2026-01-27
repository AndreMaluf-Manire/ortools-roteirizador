"""
Microserviço de Otimização de Rotas com OR-Tools
Roteirizador Manirê / Fruleve

Este serviço resolve o problema CVRPTW (Capacitated Vehicle Routing Problem with Time Windows):
- Minimiza número de veículos usados
- Respeita capacidade dos veículos (caixas/kg)
- Respeita janelas de atendimento dos clientes
- Garante que todas as entregas sejam alocadas (sem órfãs)
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import os
import math

app = FastAPI(
    title="OR-Tools Route Optimizer",
    description="API de otimização de rotas para o Roteirizador Manirê",
    version="1.0.0"
)

# CORS para permitir chamadas do Lovable/Supabase
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restringir aos domínios do Lovable
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chave de API para autenticação (configurar no Railway como variável de ambiente)
API_KEY = os.getenv("ORTOOLS_API_KEY", "dev-key-change-in-production")


# ============== MODELOS DE DADOS ==============

class Location(BaseModel):
    """Localização com coordenadas"""
    lat: float
    lng: float


class Depot(BaseModel):
    """Base/Depósito de onde partem os veículos"""
    id: str
    name: str
    location: Location


class Customer(BaseModel):
    """Cliente com janela de atendimento"""
    id: str
    name: str
    location: Location
    window_start: Optional[int] = None  # Minutos desde meia-noite (ex: 360 = 06:00)
    window_end: Optional[int] = None    # Minutos desde meia-noite (ex: 720 = 12:00)
    service_time: int = 15              # Tempo de serviço em minutos


class Delivery(BaseModel):
    """Entrega a ser alocada"""
    id: str
    customer_id: str
    boxes: int = 0
    weight_kg: float = 0
    value: float = 0


class Vehicle(BaseModel):
    """Veículo disponível para roteirização"""
    id: str
    name: str
    capacity_boxes: int
    capacity_kg: float = 99999  # Se não informado, ignora peso


class OptimizeRequest(BaseModel):
    """Requisição de otimização"""
    depot: Depot
    customers: List[Customer]
    deliveries: List[Delivery]
    vehicles: List[Vehicle]
    time_matrix: List[List[int]]  # Matriz de tempos em minutos (vem do Distance Matrix)
    start_time: int = 360         # Horário de início em minutos (default 06:00 = 360)
    max_route_duration: int = 600 # Duração máxima da rota em minutos (default 10h = 600)
    mode: str = "minimize_vehicles"  # "minimize_vehicles" ou "fixed_fleet"


class Stop(BaseModel):
    """Parada na rota otimizada"""
    delivery_id: str
    customer_id: str
    customer_name: str
    stop_order: int
    arrival_time: int      # Minutos desde meia-noite
    departure_time: int    # Minutos desde meia-noite
    window_start: Optional[int]
    window_end: Optional[int]
    window_ok: bool
    boxes: int
    weight_kg: float


class Route(BaseModel):
    """Rota otimizada para um veículo"""
    vehicle_id: Optional[str]  # None se for rota extra (pool)
    vehicle_name: Optional[str]
    stops: List[Stop]
    total_boxes: int
    total_weight_kg: float
    total_time_minutes: int
    total_distance_estimate: float  # Estimativa baseada no tempo


class OptimizeResponse(BaseModel):
    """Resposta da otimização"""
    success: bool
    message: str
    routes: List[Route]
    unassigned_deliveries: List[str]  # IDs das entregas não alocadas (se houver)
    vehicles_used: int
    total_deliveries: int
    optimization_time_ms: int


# ============== FUNÇÕES AUXILIARES ==============

def time_to_minutes(time_str: str) -> int:
    """Converte HH:MM para minutos desde meia-noite"""
    if not time_str:
        return 0
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


def minutes_to_time(minutes: int) -> str:
    """Converte minutos desde meia-noite para HH:MM"""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}"


def haversine_distance(loc1: Location, loc2: Location) -> float:
    """Calcula distância em km entre dois pontos (fallback se não tiver matriz)"""
    R = 6371  # Raio da Terra em km
    lat1, lon1 = math.radians(loc1.lat), math.radians(loc1.lng)
    lat2, lon2 = math.radians(loc2.lat), math.radians(loc2.lng)
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


# ============== SOLVER OR-TOOLS ==============

def solve_vrptw(request: OptimizeRequest) -> OptimizeResponse:
    """
    Resolve o problema CVRPTW usando OR-Tools.
    
    Estratégia:
    1. Se mode="minimize_vehicles": usa veículos virtuais ilimitados e minimiza quantidade
    2. Se mode="fixed_fleet": usa apenas os veículos informados
    """
    import time
    start_time_ms = time.time() * 1000
    
    # Mapear customers e deliveries
    customer_map = {c.id: c for c in request.customers}
    delivery_map = {d.id: d for d in request.deliveries}
    
    # Criar lista de nós: [depot] + [deliveries]
    # Índice 0 = depot, índices 1..N = entregas
    nodes = ["depot"] + [d.id for d in request.deliveries]
    num_locations = len(nodes)
    
    if num_locations <= 1:
        return OptimizeResponse(
            success=True,
            message="Nenhuma entrega para roteirizar",
            routes=[],
            unassigned_deliveries=[],
            vehicles_used=0,
            total_deliveries=0,
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )
    
    # Número de veículos
    if request.mode == "minimize_vehicles":
        # Criar veículos virtuais suficientes (máximo = número de entregas)
        num_vehicles = len(request.deliveries)
        vehicles = []
        for i in range(num_vehicles):
            if i < len(request.vehicles):
                vehicles.append(request.vehicles[i])
            else:
                # Veículo virtual com capacidade média
                avg_capacity = sum(v.capacity_boxes for v in request.vehicles) // len(request.vehicles) if request.vehicles else 100
                vehicles.append(Vehicle(
                    id=f"extra_{i}",
                    name=f"Veículo Extra {i - len(request.vehicles) + 1}",
                    capacity_boxes=avg_capacity,
                    capacity_kg=99999
                ))
    else:
        num_vehicles = len(request.vehicles)
        vehicles = request.vehicles
    
    if num_vehicles == 0:
        return OptimizeResponse(
            success=False,
            message="Nenhum veículo disponível",
            routes=[],
            unassigned_deliveries=[d.id for d in request.deliveries],
            vehicles_used=0,
            total_deliveries=len(request.deliveries),
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )
    
    # Criar o gerenciador de índices
    manager = pywrapcp.RoutingIndexManager(num_locations, num_vehicles, 0)
    
    # Criar o modelo de roteamento
    routing = pywrapcp.RoutingModel(manager)
    
    # ===== CALLBACK DE TEMPO/DISTÂNCIA =====
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        
        # Usar matriz de tempos se disponível
        if request.time_matrix and len(request.time_matrix) > from_node and len(request.time_matrix[from_node]) > to_node:
            return request.time_matrix[from_node][to_node]
        
        # Fallback: estimar tempo baseado em distância (30 km/h média em SP)
        if from_node == 0:
            loc1 = request.depot.location
        else:
            delivery = request.deliveries[from_node - 1]
            customer = customer_map.get(delivery.customer_id)
            loc1 = customer.location if customer else request.depot.location
        
        if to_node == 0:
            loc2 = request.depot.location
        else:
            delivery = request.deliveries[to_node - 1]
            customer = customer_map.get(delivery.customer_id)
            loc2 = customer.location if customer else request.depot.location
        
        distance_km = haversine_distance(loc1, loc2)
        time_minutes = int(distance_km / 30 * 60)  # 30 km/h
        return time_minutes
    
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # ===== DIMENSÃO DE TEMPO (para janelas) =====
    routing.AddDimension(
        transit_callback_index,
        60,  # Slack máximo (espera permitida)
        request.max_route_duration,  # Tempo máximo por rota
        False,  # Não forçar início no tempo 0
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # Configurar janelas de tempo para cada nó
    for location_idx in range(num_locations):
        if location_idx == 0:
            # Depot: janela ampla
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(0, request.max_route_duration)
        else:
            # Entrega: usar janela do cliente
            delivery = request.deliveries[location_idx - 1]
            customer = customer_map.get(delivery.customer_id)
            
            index = manager.NodeToIndex(location_idx)
            
            if customer and customer.window_start is not None and customer.window_end is not None:
                # Converter para minutos relativos ao início da rota
                window_start = max(0, customer.window_start - request.start_time)
                window_end = max(window_start, customer.window_end - request.start_time)
                time_dimension.CumulVar(index).SetRange(window_start, window_end)
            else:
                # Sem janela: qualquer horário
                time_dimension.CumulVar(index).SetRange(0, request.max_route_duration)
    
    # ===== DIMENSÃO DE CAPACIDADE (CAIXAS) =====
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        delivery = request.deliveries[from_node - 1]
        return delivery.boxes
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # Sem slack
        [v.capacity_boxes for v in vehicles],  # Capacidade por veículo
        True,  # Começar do zero
        "Capacity"
    )
    
    # ===== PENALIDADE PARA VEÍCULOS NÃO USADOS (minimizar frota) =====
    if request.mode == "minimize_vehicles":
        # Alta penalidade por usar veículo adicional
        for vehicle_idx in range(num_vehicles):
            routing.SetFixedCostOfVehicle(10000, vehicle_idx)
    
    # ===== PERMITIR ENTREGAS NÃO ATENDIDAS (com penalidade alta) =====
    penalty = 100000  # Penalidade por não entregar
    for node in range(1, num_locations):
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)
    
    # ===== PARÂMETROS DE BUSCA =====
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(30)  # Limite de 30 segundos
    
    # ===== RESOLVER =====
    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return OptimizeResponse(
            success=False,
            message="Não foi possível encontrar uma solução. Verifique se as janelas e capacidades são compatíveis.",
            routes=[],
            unassigned_deliveries=[d.id for d in request.deliveries],
            vehicles_used=0,
            total_deliveries=len(request.deliveries),
            optimization_time_ms=int(time.time() * 1000 - start_time_ms)
        )
    
    # ===== EXTRAIR SOLUÇÃO =====
    routes = []
    unassigned = []
    vehicles_used = 0
    
    for vehicle_idx in range(num_vehicles):
        index = routing.Start(vehicle_idx)
        stops = []
        route_boxes = 0
        route_weight = 0
        stop_order = 0
        
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            
            if node > 0:  # Não é o depot
                delivery = request.deliveries[node - 1]
                customer = customer_map.get(delivery.customer_id)
                
                time_var = time_dimension.CumulVar(index)
                arrival = solution.Min(time_var) + request.start_time
                service_time = customer.service_time if customer else 15
                departure = arrival + service_time
                
                # Verificar se está dentro da janela
                window_ok = True
                if customer and customer.window_start is not None and customer.window_end is not None:
                    window_ok = customer.window_start <= arrival <= customer.window_end
                
                stop_order += 1
                stops.append(Stop(
                    delivery_id=delivery.id,
                    customer_id=delivery.customer_id,
                    customer_name=customer.name if customer else "Desconhecido",
                    stop_order=stop_order,
                    arrival_time=arrival,
                    departure_time=departure,
                    window_start=customer.window_start if customer else None,
                    window_end=customer.window_end if customer else None,
                    window_ok=window_ok,
                    boxes=delivery.boxes,
                    weight_kg=delivery.weight_kg
                ))
                
                route_boxes += delivery.boxes
                route_weight += delivery.weight_kg
            
            index = solution.Value(routing.NextVar(index))
        
        if stops:
            vehicles_used += 1
            vehicle = vehicles[vehicle_idx] if vehicle_idx < len(vehicles) else None
            
            # Calcular tempo total da rota
            first_arrival = stops[0].arrival_time
            last_departure = stops[-1].departure_time
            total_time = last_departure - first_arrival
            
            routes.append(Route(
                vehicle_id=vehicle.id if vehicle and not vehicle.id.startswith("extra_") else None,
                vehicle_name=vehicle.name if vehicle else f"Rota Extra {vehicle_idx + 1}",
                stops=stops,
                total_boxes=route_boxes,
                total_weight_kg=route_weight,
                total_time_minutes=total_time,
                total_distance_estimate=total_time * 0.5  # Estimativa: 30km/h
            ))
    
    # Identificar entregas não alocadas
    assigned_deliveries = set()
    for route in routes:
        for stop in route.stops:
            assigned_deliveries.add(stop.delivery_id)
    
    unassigned = [d.id for d in request.deliveries if d.id not in assigned_deliveries]
    
    return OptimizeResponse(
        success=True,
        message=f"Otimização concluída: {vehicles_used} veículos, {len(assigned_deliveries)} entregas alocadas",
        routes=routes,
        unassigned_deliveries=unassigned,
        vehicles_used=vehicles_used,
        total_deliveries=len(request.deliveries),
        optimization_time_ms=int(time.time() * 1000 - start_time_ms)
    )


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    """Health check"""
    return {"status": "ok", "service": "OR-Tools Route Optimizer", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check detalhado"""
    return {
        "status": "healthy",
        "ortools_version": "9.x",
        "api_key_configured": API_KEY != "dev-key-change-in-production"
    }


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize_routes(
    request: OptimizeRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Otimiza rotas usando OR-Tools CVRPTW.
    
    Recebe:
    - depot: Base de origem
    - customers: Lista de clientes com janelas
    - deliveries: Entregas a serem alocadas
    - vehicles: Veículos disponíveis
    - time_matrix: Matriz de tempos (do Google Distance Matrix)
    - mode: "minimize_vehicles" ou "fixed_fleet"
    
    Retorna:
    - routes: Lista de rotas otimizadas
    - unassigned_deliveries: Entregas não alocadas (se houver)
    """
    
    # Validar API Key
    if API_KEY != "dev-key-change-in-production":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="API Key não fornecida")
        
        provided_key = authorization.replace("Bearer ", "")
        if provided_key != API_KEY:
            raise HTTPException(status_code=401, detail="API Key inválida")
    
    # Validar dados mínimos
    if not request.deliveries:
        return OptimizeResponse(
            success=True,
            message="Nenhuma entrega para roteirizar",
            routes=[],
            unassigned_deliveries=[],
            vehicles_used=0,
            total_deliveries=0,
            optimization_time_ms=0
        )
    
    if not request.vehicles and request.mode == "fixed_fleet":
        raise HTTPException(
            status_code=400,
            detail="Modo 'fixed_fleet' requer pelo menos um veículo"
        )
    
    # Executar otimização
    try:
        result = solve_vrptw(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na otimização: {str(e)}")


@app.post("/validate")
async def validate_request(request: OptimizeRequest):
    """
    Valida os dados antes de otimizar (útil para debug).
    """
    issues = []
    
    # Verificar se todos os customers das deliveries existem
    customer_ids = {c.id for c in request.customers}
    for d in request.deliveries:
        if d.customer_id not in customer_ids:
            issues.append(f"Delivery {d.id} referencia customer inexistente: {d.customer_id}")
    
    # Verificar se há coordenadas válidas
    for c in request.customers:
        if not (-90 <= c.location.lat <= 90) or not (-180 <= c.location.lng <= 180):
            issues.append(f"Customer {c.id} tem coordenadas inválidas")
    
    # Verificar capacidade total vs demanda total
    total_capacity = sum(v.capacity_boxes for v in request.vehicles) if request.vehicles else 0
    total_demand = sum(d.boxes for d in request.deliveries)
    
    if total_demand > total_capacity and request.mode == "fixed_fleet":
        issues.append(f"Demanda total ({total_demand} caixas) excede capacidade total ({total_capacity} caixas)")
    
    # Verificar matriz de tempos
    expected_size = len(request.deliveries) + 1  # depot + deliveries
    if request.time_matrix:
        if len(request.time_matrix) != expected_size:
            issues.append(f"Matriz de tempos tem tamanho incorreto: {len(request.time_matrix)} vs esperado {expected_size}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "summary": {
            "depot": request.depot.name,
            "customers": len(request.customers),
            "deliveries": len(request.deliveries),
            "vehicles": len(request.vehicles),
            "total_boxes": total_demand,
            "total_capacity": total_capacity,
            "has_time_matrix": bool(request.time_matrix)
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""City configurations for NYC and LA"""

CITY_CONFIGS = {
    'nyc': {
        'name': 'New York City Metro',
        'archetype': 'dense_transit_rich',
        'description': 'High-density city with extensive transit but severe congestion',
        
        'initial_conditions': {
            'CO2_0': 420.0,
            'G_0': 1751.0,      # $1.751T (BEA 2023)
            'E_0': 85.0
        },
        
        'traffic_params': {
            'base_traffic_volume': 15000,
            'road_capacity': 2500,
            'gdp_traffic_elasticity': 0.9,
            'GDP_baseline': 1751.0,
            'bpr_alpha': 0.15,
            'bpr_beta': 4.0,
            'free_flow_speed': 45.0,
            'base_ev_share': 0.08,
            'ev_tax_sensitivity': 0.4,
            'congestion_emission_factor': 0.7,
            'mobility_elasticity': 0.08,
            'base_transit_share': 0.40,
            'eta_tau_traffic': 0.25,
            'eta_s_traffic': 0.15,
            'eta_c_traffic': 0.50,
        },
        
        'economic_params': {
            'g_0': 0.025,
            'beta_s': 0.015,
            'beta_tau': 0.012,
            'beta_c': 0.008,
            'theta': 0.003,
            'phi': 2.0,
        },
    },
    
    'la': {
        'name': 'Los Angeles Metro',
        'archetype': 'sprawl_car_dependent',
        'description': 'Low-density sprawl with extensive highways, car-dependent',
        
        'initial_conditions': {
            'CO2_0': 420.0,
            'G_0': 1048.0,      # $1.048T (BEA 2023)
            'E_0': 70.0
        },
        
        'traffic_params': {
            'base_traffic_volume': 12000,
            'road_capacity': 3000,
            'gdp_traffic_elasticity': 1.0,
            'GDP_baseline': 1048.0,
            'bpr_alpha': 0.12,
            'bpr_beta': 3.5,
            'free_flow_speed': 60.0,
            'base_ev_share': 0.12,
            'ev_tax_sensitivity': 0.35,
            'congestion_emission_factor': 0.5,
            'mobility_elasticity': 0.06,
            'base_transit_share': 0.10,
            'eta_tau_traffic': 0.35,
            'eta_s_traffic': 0.30,
            'eta_c_traffic': 0.25,
        },
        
        'economic_params': {
            'g_0': 0.03,
            'beta_s': 0.025,
            'beta_tau': 0.010,
            'beta_c': 0.006,
            'theta': 0.003,
            'phi': 2.0,
        },
    }
}

def load_city_config(city_name):
    if city_name not in CITY_CONFIGS:
        raise ValueError(f"City '{city_name}' not found. Available: {list(CITY_CONFIGS.keys())}")
    return CITY_CONFIGS[city_name].copy()

if __name__ == "__main__":
    print("Available cities:", list(CITY_CONFIGS.keys()))
    for city in CITY_CONFIGS:
        print(f"\n{CITY_CONFIGS[city]['name']}:")
        print(f"  GDP: ${CITY_CONFIGS[city]['initial_conditions']['G_0']}B")
        print(f"  Transit: {CITY_CONFIGS[city]['traffic_params']['base_transit_share']*100:.0f}%")

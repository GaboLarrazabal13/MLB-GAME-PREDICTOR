"""
Script de prueba completo para la API MLB Predictor
Ejecutar: python test_api.py
"""

import requests
import json
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Cambia esto según dónde esté tu API
API_URL = "http://localhost:8000"  # Local
# API_URL = "https://tu-api.onrender.com"  # En la nube

# ============================================================================
# FUNCIONES DE PRUEBA
# ============================================================================

def test_root():
    """Test del endpoint raíz"""
    print("\n" + "="*70)
    print("TEST 1: Endpoint Raíz (GET /)")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ ERROR: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: No se pudo conectar a la API")
        print("   ¿Está la API corriendo?")
        print(f"   Verifica: {API_URL}")
        return False
    
    return True


def test_health():
    """Test del health check"""
    print("\n" + "="*70)
    print("TEST 2: Health Check (GET /health)")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS")
            print(f"Status: {data.get('status')}")
            print(f"Modelo Cargado: {data.get('model_loaded')}")
        else:
            print(f"❌ ERROR: {response.status_code}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    return True


def test_info():
    """Test del endpoint de información del modelo"""
    print("\n" + "="*70)
    print("TEST 3: Información del Modelo (GET /info)")
    print("="*70)
    
    try:
        response = requests.get(f"{API_URL}/info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS")
            print(f"\n📊 Información del Modelo:")
            print(f"   Nombre: {data.get('nombre')}")
            print(f"   Accuracy: {data.get('accuracy')*100:.2f}%")
            print(f"   ROC-AUC: {data.get('roc_auc'):.4f}")
            print(f"   Features: {data.get('n_features')}")
            print(f"   Train size: {data.get('n_train')}")
            print(f"   Test size: {data.get('n_test')}")
        else:
            print(f"❌ ERROR: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    return True


def test_prediction(home_team, away_team, home_pitcher, away_pitcher, year=2025):
    """Test de predicción"""
    print("\n" + "="*70)
    print(f"TEST 4: Predicción (POST /predict)")
    print("="*70)
    print(f"Partido: {home_team} vs {away_team}")
    print(f"Lanzadores: {home_pitcher} vs {away_pitcher}")
    
    data = {
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher": home_pitcher,
        "away_pitcher": away_pitcher,
        "year": year
    }
    
    print(f"\n📤 Enviando request...")
    
    try:
        start_time = datetime.now()
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=60  # 60 segundos de timeout
        )
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Status Code: {response.status_code}")
        print(f"Tiempo de respuesta: {duration:.2f} segundos")
        
        if response.status_code == 200:
            resultado = response.json()
            print("\n✅ SUCCESS - Predicción realizada")
            print(f"\n🏆 RESULTADO:")
            print(f"   Ganador Predicho: {resultado.get('ganador')}")
            print(f"   Probabilidad Local: {resultado.get('prob_home')*100:.1f}%")
            print(f"   Probabilidad Visitante: {resultado.get('prob_away')*100:.1f}%")
            print(f"   Confianza: {resultado.get('confianza')*100:.1f}%")
            print(f"   Año usado: {resultado.get('year_usado')}")
            
            if resultado.get('mensaje'):
                print(f"   ℹ️  {resultado.get('mensaje')}")
                
        elif response.status_code == 400:
            print(f"\n❌ ERROR 400: Request inválido")
            print(response.json())
        elif response.status_code == 503:
            print(f"\n❌ ERROR 503: Modelo no cargado")
            print(response.json())
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("❌ ERROR: Timeout - La API tardó demasiado en responder")
        print("   El scraping puede estar tomando mucho tiempo")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    
    return True


def test_invalid_prediction():
    """Test con datos inválidos"""
    print("\n" + "="*70)
    print("TEST 5: Predicción con Datos Inválidos")
    print("="*70)
    
    data = {
        "home_team": "INVALID",
        "away_team": "WRONG",
        "home_pitcher": "NoExiste",
        "away_pitcher": "Tampoco",
        "year": 2025
    }
    
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 400:
            print("✅ SUCCESS - Error manejado correctamente")
            print(f"Mensaje: {response.json().get('detail')}")
        else:
            print(f"Respuesta: {response.json()}")
            
    except Exception as e:
        print(f"Error: {e}")


def run_all_tests():
    """Ejecuta todos los tests"""
    print("\n" + "="*70)
    print(" 🧪 SUITE DE PRUEBAS - API MLB PREDICTOR")
    print("="*70)
    print(f"API URL: {API_URL}")
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Root
    if not test_root():
        print("\n❌ ABORTANDO: La API no está accesible")
        return
    
    # Test 2: Health
    test_health()
    
    # Test 3: Info
    test_info()
    
    # Test 4: Predicción válida (BOS vs NYY)
    test_prediction("BOS", "NYY", "Bello", "Cole", 2025)
    
    # Test 5: Otra predicción (LAD vs SFG)
    test_prediction("LAD", "SFG", "Yamamoto", "Webb", 2025)
    
    # Test 6: Predicción con datos inválidos
    test_invalid_prediction()
    
    # Resumen
    print("\n" + "="*70)
    print(" ✅ TESTS COMPLETADOS")
    print("="*70)
    print("\n💡 Próximos pasos:")
    print("   1. Si todos los tests pasaron, tu API funciona correctamente")
    print("   2. Puedes acceder a la documentación en:")
    print(f"      {API_URL}/docs (Swagger UI)")
    print(f"      {API_URL}/redoc (ReDoc)")
    print("   3. Para deploy en producción, consulta el README.md")


# ============================================================================
# MENÚ INTERACTIVO
# ============================================================================

def menu_interactivo():
    """Menú interactivo para pruebas personalizadas"""
    while True:
        print("\n" + "="*70)
        print(" 🎮 MENÚ INTERACTIVO - API MLB PREDICTOR")
        print("="*70)
        print("\n1. Ejecutar todos los tests")
        print("2. Probar predicción personalizada")
        print("3. Health check rápido")
        print("4. Información del modelo")
        print("5. Cambiar URL de API")
        print("6. Salir")
        
        opcion = input("\nSelecciona una opción (1-6): ").strip()
        
        if opcion == "1":
            run_all_tests()
        
        elif opcion == "2":
            print("\n📝 Ingresa los datos del partido:")
            home = input("Equipo Local (ej: BOS): ").strip().upper()
            away = input("Equipo Visitante (ej: NYY): ").strip().upper()
            pitcher_home = input("Lanzador Local (ej: Bello): ").strip()
            pitcher_away = input("Lanzador Visitante (ej: Cole): ").strip()
            year = input("Año (default 2025): ").strip()
            year = int(year) if year else 2025
            
            test_prediction(home, away, pitcher_home, pitcher_away, year)
        
        elif opcion == "3":
            test_health()
        
        elif opcion == "4":
            test_info()
        
        elif opcion == "5":
            global API_URL
            nueva_url = input(f"\nURL actual: {API_URL}\nNueva URL: ").strip()
            if nueva_url:
                API_URL = nueva_url
                print(f"✅ URL actualizada: {API_URL}")
        
        elif opcion == "6":
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("\n❌ Opción inválida")


# ============================================================================
# EJECUCIÓN
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║             API MLB PREDICTOR - TEST SUITE                    ║
    ║                                                                   ║
    ║   Script de prueba para verificar funcionamiento de la API       ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # Verificar argumentos
    if len(sys.argv) > 1:
        if sys.argv[1] == "--auto":
            # Modo automático
            run_all_tests()
        elif sys.argv[1].startswith("--url="):
            # Cambiar URL
            API_URL = sys.argv[1].split("=")[1]
            print(f"🔗 Usando URL: {API_URL}")
            run_all_tests()
        else:
            print("Uso:")
            print("  python test_api.py           # Modo interactivo")
            print("  python test_api.py --auto    # Ejecutar todos los tests")
            print("  python test_api.py --url=http://tu-api.com  # URL personalizada")
    else:
        # Modo interactivo por defecto
        menu_interactivo()
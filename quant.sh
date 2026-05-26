#!/usr/bin/env bash
# ==============================================================================
# QUANT ENGINE V2 — SCRIPT DE INSTALACIÓN Y CONTROL TOTAL (Fedora Nativo)
# Uso para instalación: sudo ./quant.sh install
# Uso para arrancar:    ./quant.sh start
# ==============================================================================

# Detener el script inmediatamente si ocurre un error inesperado
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO] $(date '+%Y-%m-%d %H:%M:%S') — $1${NC}"; }
log_warn() { echo -e "${YELLOW}[WARN] $(date '+%Y-%m-%d %H:%M:%S') — $1${NC}"; }
log_error() { echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S') — $1${NC}"; }

# Verificar que comandos del sistema se corran con privilegios sudo
check_sudo() {
    if [ "$EUID" -ne 0 ]; then
        log_error "Este comando requiere privilegios de administrador (sudo)."
        echo "Por favor, ejecuta: sudo ./quant.sh $1"
        exit 1
    fi
}

show_help() {
    echo "Uso: ./quant.sh [comando]"
    echo ""
    echo "Comandos disponibles:"
    echo "  install    INSTALACIÓN TOTAL: Detecta e instala Docker, compiladores C nativos y sintoniza el Kernel"
    echo "  start      Compila y levanta TODO el ecosistema unificado (Bot + Dashboard) localmente en segundo plano"
    echo "  stop       Apaga todos los contenedores de forma segura"
    echo "  recover    Vigilante de salud: reinicia de emergencia el core del bot si detecta una caída"
    echo "  logs       Muestra la telemetría y decisiones asíncronas del bot en tiempo real"
    echo "  status     Muestra el consumo de memoria RAM y CPU de los contenedores en vivo"
}

case "$1" in
    install)
        check_sudo "install"
        log_info "Iniciando protocolo de aprovisionamiento total para Fedora Linux..."

        # 1. Instalar herramientas de desarrollo en Fedora (para compilar NumPy y SciPy)
        log_info "Instalando herramientas de desarrollo nativas (gcc, C++, libpq)..."
        dnf install -y @development-tools @c-development || log_warn "Verifica tus repositorios dnf."
        dnf install -y libpq-devel git || log_warn "No se pudieron instalar librerías secundarias."

        # 2. Detectar e instalar Docker + Docker Compose v2 en Fedora
        if ! command -v docker &> /dev/null; then
            log_warn "Docker no detectado en Fedora. Instalando desde repositorios oficiales..."
            dnf install -y dnf-plugins-core
            dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
            dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
            
            log_info "Habilitando e iniciando servicios de Docker..."
            systemctl enable --now docker
            systemctl enable --now containerd
        else
            log_info "Docker ya se encuentra instalado correctamente en el sistema."
        fi

        # Agregar el usuario real al grupo docker para operar los comandos 'start/stop' sin requerir sudo
        REAL_USER=${SUDO_USER:-$USER}
        if [ "$REAL_USER" != "root" ]; then
            log_info "Asignando al usuario '$REAL_USER' al grupo de seguridad de docker..."
            # Asegurar que el grupo existe en Fedora
            groupadd -f docker
            usermod -aG docker "$REAL_USER"
            log_warn "¡Acción requerida!: Para activar los permisos de Docker en tu sesión, ejecuta 'newgrp docker' o reinicia la terminal."
        fi

        # 3. Optimización del Kernel de Linux (Sintonización de buffers para flujos masivos de WebSockets)
        log_info "Inyectando parámetros de baja latencia y red masiva en /etc/sysctl.d/..."
        tee /etc/sysctl.d/99-quant.conf <<'EOF'
# Evita pérdida de ticks de red en ráfagas de alta frecuencia de Binance/Alpaca
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.ipv4.tcp_keepalive_time=60
net.ipv4.tcp_keepalive_intvl=10
net.ipv4.tcp_keepalive_probes=3
# Requerido por Redis para la persistencia atómica de Sorted Sets de correlación
vm.swappiness=10
vm.overcommit_memory=1
EOF
        sysctl --system

        # 4. Verificar existencia de las variables de entorno
        if [ ! -f .env ]; then
            log_warn "No se encontró el archivo .env activo. Duplicando plantilla base..."
            cp .env.example .env
            chmod 600 .env
            log_info "Plantilla .env creada con éxito. RELLÉNALA antes de ejecutar './quant.sh start'."
        else
            chmod 600 .env
        fi

        log_info "PROCESO DE INSTALACIÓN TOTAL COMPLETADO CON ÉXITO."
        log_warn "Pasos finales obligatorios:"
        echo "  1. Ejecuta el comando: newgrp docker"
        echo "  2. Rellena tus API keys de simulación dentro del archivo .env"
        echo "  3. Arranca el motor ejecutando: ./quant.sh start"
        ;;

    start)
        log_info "Compilando y desplegando entorno unificado de pruebas v2 (Core + UI)..."
        if [ ! -f .env ]; then
            log_error "Falta el archivo .env configurado en la raíz del proyecto. No se puede iniciar."
            exit 1
        fi

        # --build:          compila SciPy/NumPy con las optimizaciones del procesador local
        # --force-recreate: descarta cualquier contenedor previo en estado inconsistente (ej. QuestDB unhealthy)
        #                   sin esta bandera, Docker reutiliza contenedores con estado roto entre reinicios
        if docker compose up -d --build --force-recreate; then
            log_info "Ecosistema completo operando localmente de forma exitosa en segundo plano."
            log_info "Acceso al Dashboard: http://localhost:8501"
            log_info "Acceso a QuestDB (SQL): http://localhost:9000"
        else
            log_error "Fallo al iniciar uno o más contenedores. Consultando logs de los servicios con errores..."
            echo ""
            docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Health}}"
            echo ""
            log_warn "Para ver el log detallado de QuestDB ejecuta: docker logs quant_questdb --tail 50"
            log_warn "Para ver todos los logs ejecuta:             ./quant.sh logs"
            exit 1
        fi
        ;;

    stop)
        log_info "Apagando el sistema unificado de forma controlada..."
        docker compose down
        log_warn "Contenedores apagados. Las órdenes de protección Bracket y OCO siguen activas en el servidor del exchange."
        ;;

    recover)
        if ! docker ps --format '{{.Names}}' | grep -q "quant_bot"; then
            log_error "¡Alerta catastrófica! El motor cuantitativo no está activo."
            log_warn "Ejecutando protocolo de recuperación y reinicio de emergencia limpio..."
            docker compose restart bot
            log_info "Bucle de concurrencia asíncrona restaurado de forma atómica."
        else
            log_info "Constantes vitales del sistema: OPERANDO NOMINALMENTE."
        fi
        ;;

    logs)
        docker compose logs -f --tail=100 bot
        ;;

    status)
        docker stats quant_bot quant_dashboard quant_questdb quant_postgres quant_redis
        ;;

    *)
        show_help
        exit 1
        ;;
esac

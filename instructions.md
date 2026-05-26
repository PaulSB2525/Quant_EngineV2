# Quant Engine — Manual de Operaciones

Este documento cubre el despliegue completo del sistema en la arquitectura
de dos nodos. **Léelo de punta a punta antes de tocar capital real.**

---

## 0. Arquitectura

```
┌─────────────────────────────────┐         ┌──────────────────────────┐
│  Node A — The Engine            │         │  Node B — The Gatekeeper │
│  Ryzen 3 / 8 GB / SSD / Linux   │  ◀────  │  Raspberry Pi 4 / 4 GB   │
│  - bot_core (asyncio)           │  LAN    │  - dashboard (Streamlit) │
│  - QuestDB + Postgres + Redis   │         │  - cloudflared tunnel    │
│  - LLM bridge                   │         │  - heartbeat monitor     │
└─────────────────────────────────┘         └──────────────────────────┘
                                                       │
                                                       ▼
                                                 Cloudflare Zero Trust
                                                       │
                                                       ▼
                                                  Tu navegador
```

- **Node A** corre todo el stack pesado (Docker compose).
- **Node B** corre solo el dashboard + el túnel de acceso remoto. Es el único
  punto que toca internet pública.
- La comunicación entre nodos es **LAN privada** (no exponer Postgres/Redis a internet).

---

## 1. Preparación del Node A (Linux Headless)

### 1.1 SO base

Probado sobre Ubuntu 22.04 LTS Server o Debian 12.

```bash
# Usuario y sudo
sudo adduser quant
sudo usermod -aG sudo quant
sudo usermod -aG docker quant   # tras instalar docker

# Hardening básico
sudo apt update && sudo apt upgrade -y
sudo apt install -y ufw fail2ban unattended-upgrades htop iotop
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.168.0.0/16 to any port 22  # SSH solo desde LAN
sudo ufw allow from 192.168.0.0/16 to any port 5432
sudo ufw allow from 192.168.0.0/16 to any port 6379
sudo ufw enable
```

### 1.2 Docker

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo systemctl enable --now docker

# Verifica
docker --version
docker compose version
```

### 1.3 Tuning del kernel (latencia)

```bash
# /etc/sysctl.d/99-quant.conf
sudo tee /etc/sysctl.d/99-quant.conf <<'EOF'
# Redes (websocket largos)
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.ipv4.tcp_keepalive_time=60
net.ipv4.tcp_keepalive_intvl=10
net.ipv4.tcp_keepalive_probes=3
# VM
vm.swappiness=10
vm.overcommit_memory=1   # requerido por Redis
EOF
sudo sysctl --system
```

### 1.4 SSD tuning

```bash
# Asegúrate de que tu SSD use el scheduler 'none' o 'mq-deadline'
cat /sys/block/sda/queue/scheduler   # debería incluir [none] o [mq-deadline]
# Y monta con noatime:
# Edita /etc/fstab:  defaults,noatime,nodiratime
```

---

## 2. Clonado y configuración

```bash
cd ~
git clone <tu-repo>.git quant
cd quant

cp .env.example .env
nano .env   # rellena las variables (sección §4)
```

---

## 3. Generación de credenciales

### 3.1 API del Exchange (Binance recomendado para spot, OKX para futuros)

1. Crea **subcuenta** dedicada al bot. Nunca uses la cuenta principal.
2. Genera API key con:
   - ✅ Read
   - ✅ Spot & Margin Trading
   - ❌ Withdrawals (NUNCA)
3. **Whitelist tu IP estática** del Node A. Si no tienes IP estática, usa DDNS.
4. Copia `EXCHANGE_API_KEY` y `EXCHANGE_API_SECRET` al `.env`.

### 3.2 LLMs

- **Gemini**: https://aistudio.google.com → "Get API key" → `GEMINI_API_KEY`.
- **DeepSeek**: https://platform.deepseek.com → "API Keys" → `DEEPSEEK_API_KEY`.

### 3.3 Telegram

```
# 1. Habla con @BotFather en Telegram, crea un bot, copia el token => TELEGRAM_BOT_TOKEN
# 2. Habla con @userinfobot, copia tu ID => TELEGRAM_CHAT_ID
# 3. Envía /start a tu bot (sin esto, no puede mandarte mensajes)
```

### 3.4 Hash de password del dashboard

En tu laptop (no en producción):

```bash
python3 -c "import streamlit_authenticator as sa; print(sa.Hasher.hash('MiPasswordFuerte'))"
# Pega el output en ADMIN_PASSWORD_HASH dentro del .env del Node B
```

Genera también un `AUTH_COOKIE_KEY` aleatorio:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## 4. Variables de entorno (`.env`)

| Variable                   | Descripción                                                |
|----------------------------|------------------------------------------------------------|
| `EXCHANGE_ID`              | `binance`, `okx`, `bybit`, etc. (ccxt id).                |
| `EXCHANGE_API_KEY/SECRET`  | Credenciales con whitelist de IP.                          |
| `PAPER_TRADING`            | `true` para sandbox. **Empieza siempre así.**             |
| `SYMBOLS`                  | Coma-separados. `BTC/USDT,ETH/USDT,SOL/USDT`.             |
| `QUOTE_CURRENCY`           | Moneda en la que reportar equity (`USDT` típico).         |
| `POSTGRES_USER/PASSWORD/DB`| Para el contenedor de Postgres.                            |
| `TELEGRAM_BOT_TOKEN/CHAT_ID`| Alertas.                                                  |
| `ENABLE_LLM_VALIDATION`    | `true` para que DeepSeek valide trades de alpha modesto.  |
| `GEMINI_API_KEY`           | Análisis de sentimiento de noticias.                      |
| `DEEPSEEK_API_KEY`         | Validación de tesis de trading.                            |
| `ADMIN_PASSWORD_HASH`      | bcrypt-hash del password del dashboard.                    |
| `AUTH_COOKIE_KEY`          | ≥32 chars random para firmar la cookie.                    |

> **Permisos del `.env`**:
> ```bash
> chmod 600 .env
> chown quant:quant .env
> ```

---

## 5. Despliegue en Node A

```bash
cd ~/quant
docker compose up -d

# Verifica
docker compose ps
docker compose logs -f bot       # logs del bot en vivo
docker compose logs questdb postgres redis
```

**Primer arranque esperado:**

- QuestDB tarda ~10s en levantar.
- Postgres ~5s y aplica el schema automáticamente al primer connect del bot.
- Redis es instantáneo.
- El bot empieza a publicar al websocket y a llenar el buffer OU/GARCH.
- Verás logs `GARCH refit BTC/USDT: alpha=… beta=… persist=…` después de ~25 min
  (necesita 500 retornos de tick-segundo).

### 5.1 Logs estructurados

```bash
# Stream
docker compose logs -f --tail=200 bot

# Buscar decisiones rechazadas por TCA en las últimas horas
docker compose logs bot --since 6h | grep "Rechazado"
```

### 5.2 Acceso directo a QuestDB

Abre `http://NODE_A_IP:9000` en tu LAN. Tienes SQL console:

```sql
SELECT timestamp, symbol, mid, kalman_mid
FROM orderbook
WHERE timestamp > dateadd('h', -1, now())
SAMPLE BY 1m;
```

---

## 6. Despliegue del Dashboard en Node B (Raspberry Pi 4)

### 6.1 SO base

Recomendado: **Raspberry Pi OS Lite (64-bit)**.

```bash
# Activar 64-bit y memoria de GPU mínima
sudo raspi-config   # Advanced > GPU Memory > 16

# Update
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl
```

### 6.2 Docker en Pi

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
# logout/login
```

### 6.3 Solo el dashboard (compose ligero)

Crea `~/quant-dashboard/docker-compose.yml` en la Pi:

```yaml
version: "3.9"
services:
  dashboard:
    image: python:3.12-slim
    container_name: dashboard
    restart: unless-stopped
    working_dir: /app
    volumes:
      - ./:/app
    env_file: .env
    environment:
      POSTGRES_DSN: postgresql://quant:changeme_strong_password@NODE_A_IP:5432/quant
      REDIS_URL: redis://NODE_A_IP:6379/0
    ports:
      - "8501:8501"
    command: >
      sh -c "apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev &&
             pip install -r requirements-dashboard.txt &&
             streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"
    deploy:
      resources:
        limits:
          memory: 768M
```

Copia `dashboard.py`, `requirements-dashboard.txt` y `.env` (con el HASH y los DSN apuntando al Node A) a `~/quant-dashboard/`.

```bash
cd ~/quant-dashboard
docker compose up -d
# Verifica en http://PI_IP:8501
```

---

## 7. Cloudflare Tunnel (Zero Trust) en la Pi

El dashboard NO debe estar expuesto en internet pública. Cloudflare Tunnel da
acceso autenticado sin abrir puertos.

### 7.1 Instalar `cloudflared`

```bash
# Pi 64-bit
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
cloudflared --version
```

### 7.2 Autenticar y crear túnel

```bash
cloudflared tunnel login
# Abre URL en tu navegador, selecciona tu dominio gestionado en Cloudflare.

cloudflared tunnel create quant-admin
# => Genera ~/.cloudflared/<UUID>.json con las credenciales del túnel.
```

### 7.3 Configurar `~/.cloudflared/config.yml`

```yaml
tunnel: <TUNNEL_UUID>
credentials-file: /home/pi/.cloudflared/<TUNNEL_UUID>.json

ingress:
  - hostname: admin.tudominio.com
    service: http://localhost:8501
  - service: http_status:404
```

### 7.4 Asociar DNS

```bash
cloudflared tunnel route dns quant-admin admin.tudominio.com
```

### 7.5 Crear política Zero Trust

En `https://one.dash.cloudflare.com`:

1. **Access → Applications → Add an application → Self-hosted**.
2. Hostname: `admin.tudominio.com`.
3. Policy: "Allow", incluye tu email (o tu grupo de IdP).
4. Session duration: 24h.

Ahora `https://admin.tudominio.com` exige autenticación de Cloudflare **antes** de
llegar al login de Streamlit (defensa en profundidad).

### 7.6 Servicio systemd

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

---

## 8. Operación diaria

### 8.1 Checklist de arranque

```
[ ] docker compose ps en Node A: todos UP, sin restarts en loop
[ ] Telegram: mensaje "🟢 Bot iniciado" recibido
[ ] Dashboard accesible y muestra equity_now correcto
[ ] Redis day_locked vacío (no es un día bloqueado por DD previo)
[ ] PAPER_TRADING=true para los primeros 14 días
```

### 8.2 Pasaje a capital real

**No pases a live** hasta cumplir TODAS:

1. ≥14 días de paper sin crashes.
2. Sharpe en ventana de paper > 1.0 (anualizado).
3. Max DD en paper < 4% (i.e. peor que el límite duro pero aceptable).
4. ≥30 trades cerrados.
5. Profit factor > 1.3.

Entonces:

```bash
# 1. En .env:  PAPER_TRADING=false
# 2. Empieza con 10% del capital target (e.g. $400 MXN si target son $4000)
docker compose down bot
docker compose up -d bot
```

### 8.3 Monitoreo diario

- 09:00 (tu zona): revisa Telegram → reporte 24h.
- Antes de dormir: dashboard → Underwater chart en `tab1`. Si DD > 1%, revisar.
- Semanal: descarga `trades` y haz post-mortem de pérdidas.

### 8.4 Apagado controlado

```bash
docker compose down bot     # bot manda mensaje "🔴 Bot detenido"
# Esto NO cierra posiciones abiertas — los SL/TP ya están en el exchange.
```

Si quieres cerrar todo:

```bash
# Manualmente desde Binance: cierra posiciones y cancela órdenes pendientes.
docker compose down
```

---

## 9. Mantenimiento

### 9.1 Backups

```bash
# Cron en Node A
0 4 * * * docker exec quant_postgres pg_dump -U quant quant | gzip > /backup/pg-$(date +\%F).sql.gz
0 4 * * * docker exec quant_redis redis-cli SAVE && cp /var/lib/docker/volumes/quant_redis_data/_data/dump.rdb /backup/
```

### 9.2 Rotación de logs

Docker ya rota (`max-size: 10m, max-file: 5` en compose). Para QuestDB:

```bash
# QuestDB no rota automáticamente. Limita retención:
docker exec quant_questdb sh -c 'find /var/lib/questdb/db/orderbook* -mtime +30 -delete'
```

### 9.3 Updates

```bash
git pull
docker compose build --no-cache bot dashboard
docker compose up -d
```

---

## 10. Troubleshooting

| Síntoma                                 | Causa probable                       | Solución |
|-----------------------------------------|--------------------------------------|----------|
| `WS error … 1006`                       | Internet inestable                    | Bot reconecta solo con backoff. |
| `GARCH fit falló`                       | Retornos con outliers extremos       | Winsorize previo; investiga el dato. |
| `Day locked: drawdown 2.X% activado`    | El día perdió 2%                     | Esperar a 00:00 UTC. |
| Dashboard en Pi devuelve `connection refused` | Postgres bloqueado por UFW    | `sudo ufw allow from PI_IP to any port 5432` |
| Bot consume 100% CPU                    | OU refit en bucle (buffer corrupto)  | Reinicia: `docker compose restart bot` |
| Telegram no entrega                      | No has hecho `/start` al bot         | Hablar con el bot una vez. |

---

## 11. Escalado (de $4k MXN a $1M USD)

La arquitectura ES la misma; solo cambia:

| Capital | Cambios |
|---------|---------|
| ≤ $1k USD | 1 símbolo, paper o microsize. |
| $1k-$50k | 2-4 símbolos. Vol regime filter activo. |
| $50k-$500k | Considera mover el engine a un VPS dedicado cerca del exchange (Binance: AWS Tokyo). |
| > $500k | Pasa a un exchange de derivados (OKX/Bybit/dYdX), añade *colocation* y reduce `decision_period_secs` a 0.5. Implementa execution-aware models (e.g. Almgren-Chriss para slicing). |

Nada del código necesita reescribirse — los parámetros de `RiskConfig` y los
límites de exposición son los que ajustan el régimen.

---

## 12. Disclaimer

El trading algorítmico tiene riesgo real de pérdida total del capital. Este
código se entrega "as-is", sin garantías. Antes de operar capital real:

- Lee y entiende cada línea de `engine_math.py` y `risk_manager.py`.
- Audita las decisiones del bot por al menos 2 semanas en paper.
- Empieza con < 5% de tu capital de trading total.
- Define a-priori el monto máximo que puedes permitirte perder.

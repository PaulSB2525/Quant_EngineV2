-- =============================================================================
-- migrations/002_add_pending_status.sql
-- Fix C-3 / C-4 / C-5 — Patrón outbox para la tabla `trades`.
--
-- El campo `status` es TEXT sin constraint/enum, así que los NUEVOS valores no
-- requieren cambio de esquema; se documentan aquí como contrato del código:
--
--   'pending'                 -> fila escrita ANTES de enviar al broker (outbox)
--   'open'                    -> confirmado tras fill exitoso
--   'closed'                  -> round-trip cerrado (ver exit_reason)
--   'failed_unprotected'      -> entry OK pero SL/TP falló; cierre de emergencia
--   'orphaned'               -> posición abierta en exchange que NO se pudo cerrar
--   'pair_leg_a_orphaned'     -> rollback del leg A de un par falló
--   'pair_leg_b_close_failed' -> leg A cerrado, leg B no (par parcialmente abierto)
--
-- Sí se añaden dos columnas para auditar cierres y fallos:
--   exit_reason : 'tp' | 'sl' | 'timeout' | 'panic_close' | 'emergency_close'
--   notes       : texto libre (mensaje de error del broker, motivo de orphan)
--
-- Idempotente: ADD COLUMN IF NOT EXISTS. El bot también ejecuta estos ALTER en
-- el arranque (SCHEMA_SQL en bot_core.py) para DBs ya creadas.
-- =============================================================================

-- ===== UP =====
ALTER TABLE trades ADD COLUMN IF NOT EXISTS exit_reason TEXT;
ALTER TABLE trades ADD COLUMN IF NOT EXISTS notes TEXT;
CREATE INDEX IF NOT EXISTS idx_trades_status_partner ON trades(status, pair_partner);

-- ===== DOWN (rollback) =====
-- Ejecutar manualmente para revertir:
--   DROP INDEX IF EXISTS idx_trades_status_partner;
--   ALTER TABLE trades DROP COLUMN IF EXISTS notes;
--   ALTER TABLE trades DROP COLUMN IF EXISTS exit_reason;
-- NOTA: las filas con status nuevos ('pending','orphaned',...) quedarían con
-- valores que el código v1 no entiende; cerrarlas/migrarlas antes del rollback.

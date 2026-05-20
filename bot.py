import os
import json
import logging
from datetime import datetime

import httpx
from telegram import Update
from telegram.ext import Application, MessageHandler, CommandHandler, filters, ContextTypes
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Clientes ---
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

TELEGRAM_TOKEN    = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
ALLOWED_USERNAMES = set(os.environ.get("ALLOWED_USERNAMES", "").split(","))  # ej: "usuario1,usuario2"


# ── helpers ──────────────────────────────────────────────────────────────────

def usuario_autorizado(update: Update) -> bool:
    username = update.effective_user.username or ""
    return not ALLOWED_USERNAMES or username in ALLOWED_USERNAMES


async def transcribir_audio(audio_bytes: bytes, filename: str) -> str:
    """Envía el audio a Whisper y devuelve el texto."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            files={"file": (filename, audio_bytes, "audio/ogg")},
            data={"model": "whisper-1", "language": "es"},
        )
        resp.raise_for_status()
        return resp.json()["text"]


async def extraer_gasto(texto: str, nombre_usuario: str) -> dict:
    """Usa GPT-4o mini para extraer los campos del gasto desde el texto."""
    hoy = datetime.now().strftime("%d/%m/%Y")
    prompt = f"""Analiza este texto sobre un gasto y extrae la información.
Responde SOLO con JSON válido, sin texto extra, sin bloques de código.

Texto: "{texto}"
Quien lo registra: {nombre_usuario}
Fecha de hoy: {hoy}

El JSON debe tener exactamente estas claves:
- descripcion: string (nombre del gasto, máx 60 chars)
- monto: number (en pesos chilenos; "lucas"=miles, "luca"=mil, dólares×950)
- categoria: una de [comida, transporte, hogar, salud, ocio, otro]
- fecha: string DD/MM/YYYY (usa hoy si no se menciona)
- cuotas: integer (1 si es al contado)
- entre_quienes: array de strings (nombres; si dice "yo" usa "{nombre_usuario}"; si no menciona nadie usa ["{nombre_usuario}"])
- mi_parte: number (monto que le corresponde a quien registra)
- notas: string (info extra o cadena vacía)"""

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "max_tokens": 512,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": "Eres un asistente que extrae datos de gastos y responde solo con JSON válido."},
                    {"role": "user", "content": prompt},
                ],
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)


def guardar_gasto(gasto: dict, username: str) -> dict:
    """Inserta el gasto en Supabase y devuelve la fila creada."""
    # Convierte fecha DD/MM/YYYY → YYYY-MM-DD para Postgres
    try:
        fecha_obj = datetime.strptime(gasto["fecha"], "%d/%m/%Y")
        fecha_iso = fecha_obj.strftime("%Y-%m-%d")
    except Exception:
        fecha_iso = datetime.now().strftime("%Y-%m-%d")

    fila = {
        "descripcion":   gasto.get("descripcion", "Sin descripción"),
        "monto":         float(gasto.get("monto", 0)),
        "categoria":     gasto.get("categoria", "otro"),
        "fecha":         fecha_iso,
        "cuotas":        int(gasto.get("cuotas", 1)),
        "entre_quienes": gasto.get("entre_quienes", [username]),
        "mi_parte":      float(gasto.get("mi_parte", gasto.get("monto", 0))),
        "notas":         gasto.get("notas", ""),
        "registrado_por": username,
        "texto_original": gasto.get("_texto_original", ""),
    }
    result = supabase.table("gastos").insert(fila).execute()
    return result.data[0]


def formatear_confirmacion(gasto: dict, fila: dict) -> str:
    """Arma el mensaje de confirmación para Telegram."""
    monto_fmt = f"${fila['monto']:,.0f}".replace(",", ".")
    mi_parte_fmt = f"${fila['mi_parte']:,.0f}".replace(",", ".")
    personas = ", ".join(fila["entre_quienes"])
    cuota_txt = f"en {fila['cuotas']} cuotas" if fila["cuotas"] > 1 else "al contado"

    lineas = [
        "✅ *Gasto registrado*",
        f"📝 {fila['descripcion']}",
        f"💰 {monto_fmt} ({cuota_txt})",
        f"🏷️ {fila['categoria'].capitalize()}",
        f"📅 {gasto.get('fecha', '')}",
        f"👥 {personas}",
    ]
    if len(fila["entre_quienes"]) > 1:
        lineas.append(f"🔹 Tu parte: {mi_parte_fmt}")
    if fila["notas"]:
        lineas.append(f"📌 {fila['notas']}")
    return "\n".join(lineas)


# ── handlers ─────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not usuario_autorizado(update):
        await update.message.reply_text("⛔ No estás autorizado para usar este bot.")
        return
    await update.message.reply_text(
        "👋 Hola! Mándame un audio describiendo tu gasto y lo registro automáticamente.\n\n"
        "Ejemplos:\n"
        "• _'Uber 3500 pesos, lo pagué yo'_\n"
        "• _'Supermercado 52 lucas en 3 cuotas, entre los dos'_\n"
        "• _'Netflix 6 dólares, somos 4'_",
        parse_mode="Markdown",
    )


async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not usuario_autorizado(update):
        return
    # Últimos 5 gastos
    result = supabase.table("gastos").select("*").order("created_at", desc=True).limit(5).execute()
    if not result.data:
        await update.message.reply_text("📭 No hay gastos registrados aún.")
        return
    lineas = ["📊 *Últimos 5 gastos:*\n"]
    for g in result.data:
        monto = f"${g['monto']:,.0f}".replace(",", ".")
        lineas.append(f"• {g['descripcion']} — {monto} ({g['categoria']})")
    await update.message.reply_text("\n".join(lineas), parse_mode="Markdown")


async def manejar_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not usuario_autorizado(update):
        await update.message.reply_text("⛔ No estás autorizado.")
        return

    msg = await update.message.reply_text("🎙️ Procesando audio...")
    username = update.effective_user.first_name or update.effective_user.username or "Usuario"

    try:
        # 1. Descargar audio
        audio = update.message.voice or update.message.audio
        file = await context.bot.get_file(audio.file_id)
        audio_bytes = await file.download_as_bytearray()

        await msg.edit_text("📝 Transcribiendo...")

        # 2. Transcribir con Whisper
        texto = await transcribir_audio(bytes(audio_bytes), "audio.ogg")
        logger.info(f"Transcripción: {texto}")

        await msg.edit_text("🤖 Analizando gasto...")

        # 3. Extraer datos con Claude
        gasto = await extraer_gasto(texto, username)
        gasto["_texto_original"] = texto

        # 4. Guardar en Supabase
        fila = guardar_gasto(gasto, username)

        # 5. Confirmar al usuario
        await msg.edit_text(formatear_confirmacion(gasto, fila), parse_mode="Markdown")

    except json.JSONDecodeError:
        await msg.edit_text("❌ No pude interpretar el gasto. Intenta ser más específico con el monto.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await msg.edit_text(f"❌ Error inesperado: {str(e)[:100]}")


async def manejar_texto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """También acepta texto escrito (no solo audio)."""
    if not usuario_autorizado(update):
        return
    texto = update.message.text
    if texto.startswith("/"):
        return

    msg = await update.message.reply_text("🤖 Analizando gasto...")
    username = update.effective_user.first_name or update.effective_user.username or "Usuario"

    try:
        gasto = await extraer_gasto(texto, username)
        gasto["_texto_original"] = texto
        fila = guardar_gasto(gasto, username)
        await msg.edit_text(formatear_confirmacion(gasto, fila), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await msg.edit_text("❌ No pude registrar el gasto. Revisa el formato.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("resumen", cmd_resumen))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, manejar_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_texto))
    logger.info("Bot iniciado...")
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app.run_polling()


if __name__ == "__main__":
    main()

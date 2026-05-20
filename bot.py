import os
import json
import logging
from datetime import datetime

import asyncio
import base64
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import Conflict
from telegram.ext import (
    Application, MessageHandler, CommandHandler,
    CallbackQueryHandler, filters, ContextTypes
)
from telegram.request import HTTPXRequest
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Clientes ---
supabase: Client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

TELEGRAM_TOKEN    = os.environ["TELEGRAM_TOKEN"]
OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
ALLOWED_USERNAMES = set(os.environ.get("ALLOWED_USERNAMES", "").split(","))

# Almacena gastos pendientes de confirmación: {chat_id: gasto_dict}
gastos_pendientes: dict = {}


# ── helpers ───────────────────────────────────────────────────────────────────

def usuario_autorizado(update: Update) -> bool:
    username = update.effective_user.username or ""
    return not ALLOWED_USERNAMES or username in ALLOWED_USERNAMES


async def transcribir_audio(audio_bytes: bytes, filename: str) -> str:
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
    hoy = datetime.now().strftime("%d/%m/%Y")
    mes_actual = datetime.now().strftime("%m/%Y")
    prompt = f"""Analiza este texto sobre un gasto y extrae la información.
Responde SOLO con JSON válido, sin texto extra, sin bloques de código.

Texto: "{texto}"
Quien lo registra: {nombre_usuario}
Fecha de hoy: {hoy}
Mes actual: {mes_actual}

El JSON debe tener exactamente estas claves:
- descripcion: string (nombre del gasto, máx 60 chars)
- monto: number (en pesos chilenos; "lucas"=miles, "luca"=mil, dólares×950)
- categoria: una de [comida, transporte, hogar, salud, ocio, mascota, otro]. Reglas de prioridad:
    * "mascota" tiene prioridad sobre cualquier otra categoría si el gasto es relacionado con una mascota.
      Palabras clave: "Ari", "veterinario", "vet", "vacuna", "antiparasitario", "collar", "correa",
      "alimento perro", "alimento gato", "comida Ari", "snack Ari", "peluquería canina", "baño perro",
      "arena gato", "juguete perro", "juguete gato", "consulta veterinaria", "medicamento Ari", "pulgas".
      IMPORTANTE: "comida Ari" o "alimento Ari" → mascota, NO comida.
    * Si no aplica ninguna palabra clave de mascota, usar la categoría más apropiada del resto.
- fecha: string DD/MM/YYYY (usa hoy si no se menciona)
- cuotas: integer (1 si es al contado)
- fecha_primera_cuota: string MM/YYYY — mes en que cae la primera cuota. Reglas:
    * Si dice "empieza en marzo", "primer cobro en abril", "desde febrero", "la primera cae en mayo" → usar ese mes del año actual (o el mencionado)
    * Si dice "el mes que viene", "próximo mes", "mes siguiente" → sumar 1 mes al mes actual
    * Si dice "en dos meses" → sumar 2 meses al mes actual
    * Si dice "desde ya", "este mes", "cuota uno este mes" → usar el mes actual
    * Si no se menciona nada sobre cuándo empieza → usar el mes actual por defecto
    * Solo es relevante si cuotas > 1; si es al contado usar el mes actual igual
- entre_quienes: SIEMPRE ["Javier", "Romina"] — todos los gastos son compartidos entre los dos
- pagado_por: string (quien pagó; si dice "yo pagué" o no se menciona usa "{nombre_usuario}")
- mi_parte: number (SIEMPRE monto / 2 — se divide 50/50)
- notas: string (info extra o cadena vacía)"""

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
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
        return json.loads(raw.replace("```json", "").replace("```", "").strip())


async def extraer_gasto_desde_imagen(imagen_bytes: bytes, nombre_usuario: str) -> dict:
    """Usa GPT-4o vision para extraer datos de una foto de boleta."""
    hoy = datetime.now().strftime("%d/%m/%Y")
    mes_actual = datetime.now().strftime("%m/%Y")
    imagen_b64 = base64.b64encode(imagen_bytes).decode("utf-8")

    prompt = f"""Analiza esta foto de una boleta/ticket/recibo y extrae la información del gasto.
Responde SOLO con JSON válido, sin texto extra, sin bloques de código.

Quien tomó la foto (asume que pagó): {nombre_usuario}
Fecha de hoy: {hoy}
Mes actual: {mes_actual}

El JSON debe tener exactamente estas claves:
- descripcion: string (nombre del comercio o tipo de gasto, máx 60 chars)
- monto: number (monto TOTAL en pesos chilenos; busca "TOTAL", "Total a pagar", "Total $"; si ves dólares multiplica por 950)
- categoria: una de [comida, transporte, hogar, salud, ocio, mascota, otro]
- fecha: string DD/MM/YYYY (léela de la boleta; si no se ve claramente usa hoy)
- cuotas: integer (1 — asume al contado a menos que la boleta diga cuotas)
- fecha_primera_cuota: string MM/YYYY (usa el mes actual por defecto)
- entre_quienes: SIEMPRE ["Javier", "Romina"] — todos los gastos son compartidos entre los dos
- pagado_por: string ("{nombre_usuario}" — quien tomó la foto pagó)
- mi_parte: number (SIEMPRE monto / 2 — se divide 50/50)
- notas: string (si ves items relevantes en la boleta menciónalos brevemente, si no cadena vacía)
- confianza: string ("alta", "media", "baja") — qué tan legible estaba la boleta"""

    async with httpx.AsyncClient(timeout=45) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "gpt-4o",
                "max_tokens": 600,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/jpeg;base64,{imagen_b64}",
                                "detail": "low"
                            }},
                        ],
                    }
                ],
            },
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return json.loads(raw.replace("```json", "").replace("```", "").strip())


def guardar_gasto(gasto: dict, username: str) -> dict:
    try:
        fecha_obj = datetime.strptime(gasto["fecha"], "%d/%m/%Y")
        fecha_iso = fecha_obj.strftime("%Y-%m-%d")
    except Exception:
        fecha_iso = datetime.now().strftime("%Y-%m-%d")

    # Convierte fecha_primera_cuota MM/YYYY → YYYY-MM-01
    try:
        fpc_raw = gasto.get("fecha_primera_cuota", "")
        if fpc_raw:
            fpc_obj = datetime.strptime(fpc_raw, "%m/%Y")
        else:
            fpc_obj = datetime.now().replace(day=1)
        fecha_primera_cuota_iso = fpc_obj.strftime("%Y-%m-01")
    except Exception:
        fecha_primera_cuota_iso = datetime.now().strftime("%Y-%m-01")

    fila = {
        "descripcion":         gasto.get("descripcion", "Sin descripción"),
        "monto":               float(gasto.get("monto", 0)),
        "categoria":           gasto.get("categoria", "otro"),
        "fecha":               fecha_iso,
        "cuotas":              int(gasto.get("cuotas", 1)),
        "fecha_primera_cuota": fecha_primera_cuota_iso,
        "entre_quienes":       gasto.get("entre_quienes", [username]),
        "pagado_por":          gasto.get("pagado_por", username),
        "mi_parte":            float(gasto.get("mi_parte", gasto.get("monto", 0))),
        "notas":               gasto.get("notas", ""),
        "registrado_por":      username,
        "texto_original":      gasto.get("_texto_original", ""),
    }
    result = supabase.table("gastos").insert(fila).execute()
    return result.data[0]


def formatear_preview(gasto: dict, username: str) -> str:
    """Mensaje de previsualización antes de confirmar."""
    monto      = float(gasto.get("monto", 0))
    monto_fmt  = f"${monto:,.0f}".replace(",", ".")
    mi_parte   = f"${monto/2:,.0f}".replace(",", ".")
    pagado_por = gasto.get("pagado_por", username)
    n_cuotas   = int(gasto.get("cuotas", 1))
    cuota_txt  = f"en {n_cuotas} cuotas" if n_cuotas > 1 else "al contado"
    fpc        = gasto.get("fecha_primera_cuota", "")
    cuota_c_u  = f"${monto / n_cuotas / 2:,.0f}".replace(",", ".")

    lineas = [
        "🔍 *¿Confirmas este gasto?*\n",
        f"📝 {gasto.get('descripcion', '')}",
        f"💰 {monto_fmt} ({cuota_txt})",
    ]
    if n_cuotas > 1:
        lineas.append(f"📆 Primera cuota: {fpc} — {cuota_c_u}/mes c/u")
    lineas += [
        f"🏷️ {gasto.get('categoria', '').capitalize()}",
        f"📅 {gasto.get('fecha', '')}",
        f"💳 Pagó: {pagado_por}",
        f"👥 Javier & Romina — {mi_parte} c/u",
    ]
    if gasto.get("notas"):
        lineas.append(f"📌 {gasto['notas']}")
    return "\n".join(lineas)


def formatear_guardado(fila: dict) -> str:
    """Mensaje de confirmación tras guardar."""
    monto_fmt = f"${fila['monto']:,.0f}".replace(",", ".")
    cuota_txt = f"en {fila['cuotas']} cuotas" if fila["cuotas"] > 1 else "al contado"
    lineas = [
        "✅ *Gasto guardado*",
        f"📝 {fila['descripcion']} — {monto_fmt} ({cuota_txt})",
    ]
    return "\n".join(lineas)


async def procesar_y_preguntar(update: Update, context: ContextTypes.DEFAULT_TYPE, texto: str):
    """Extrae el gasto y muestra botones de confirmar/cancelar."""
    chat_id  = update.effective_chat.id
    username = update.effective_user.first_name or update.effective_user.username or "Usuario"
    msg = await update.message.reply_text("🤖 Analizando gasto...")

    try:
        gasto = await extraer_gasto(texto, username)
        gasto["_texto_original"] = texto
        gastos_pendientes[chat_id] = gasto

        teclado = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("✅ Confirmar", callback_data="confirmar"),
                InlineKeyboardButton("❌ Cancelar",  callback_data="cancelar"),
            ]
        ])
        await msg.edit_text(
            formatear_preview(gasto, username),
            parse_mode="Markdown",
            reply_markup=teclado,
        )
    except json.JSONDecodeError:
        await msg.edit_text("❌ No pude interpretar el gasto. Sé más específico con el monto.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await msg.edit_text(f"❌ Error: {str(e)[:100]}")


# ── handlers ──────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not usuario_autorizado(update):
        await update.message.reply_text("⛔ No estás autorizado.")
        return
    await update.message.reply_text(
        "👋 Hola! Mándame un audio o escribe tu gasto.\n\n"
        "Ejemplos:\n"
        "• _'Pagué 25 lucas de uber, entre Javier y Romina'_\n"
        "• _'Supermercado 52 lucas en 3 cuotas, lo pagué yo'_\n"
        "• _'Netflix 6 dólares, somos 2'_\n\n"
        "Comandos:\n"
        "/resumen — últimos 5 gastos\n"
        "/eliminar — borrar el último gasto\n"
        "/eliminar N — borrar gasto por ID",
        parse_mode="Markdown",
    )


async def cmd_resumen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not usuario_autorizado(update):
        return
    result = supabase.table("gastos").select("*").order("created_at", desc=True).limit(5).execute()
    if not result.data:
        await update.message.reply_text("📭 No hay gastos registrados aún.")
        return
    lineas = ["📊 *Últimos 5 gastos:*\n"]
    for g in result.data:
        monto = f"${g['monto']:,.0f}".replace(",", ".")
        lineas.append(f"• `#{g['id']}` {g['descripcion']} — {monto}")
    await update.message.reply_text("\n".join(lineas), parse_mode="Markdown")


async def cmd_eliminar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not usuario_autorizado(update):
        return

    # Si se pasa un ID: /eliminar 42
    args = context.args
    if args:
        try:
            gasto_id = int(args[0])
        except ValueError:
            await update.message.reply_text("Uso: /eliminar o /eliminar ID")
            return
        result = supabase.table("gastos").select("*").eq("id", gasto_id).execute()
        if not result.data:
            await update.message.reply_text(f"❌ No encontré el gasto #{gasto_id}.")
            return
        g = result.data[0]
    else:
        # Sin ID: eliminar el último gasto
        result = supabase.table("gastos").select("*").order("created_at", desc=True).limit(1).execute()
        if not result.data:
            await update.message.reply_text("📭 No hay gastos para eliminar.")
            return
        g = result.data[0]

    monto = f"${g['monto']:,.0f}".replace(",", ".")
    context.user_data["eliminar_id"] = g["id"]

    teclado = InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🗑️ Sí, eliminar", callback_data=f"eliminar_{g['id']}"),
            InlineKeyboardButton("Cancelar",        callback_data="cancelar_eliminar"),
        ]
    ])
    await update.message.reply_text(
        f"¿Eliminar este gasto?\n\n`#{g['id']}` *{g['descripcion']}* — {monto}\n📅 {g['fecha']}",
        parse_mode="Markdown",
        reply_markup=teclado,
    )


async def manejar_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id  = update.effective_chat.id
    username = update.effective_user.first_name or update.effective_user.username or "Usuario"
    data     = query.data

    if data == "confirmar":
        gasto = gastos_pendientes.pop(chat_id, None)
        if not gasto:
            await query.edit_message_text("⚠️ No hay gasto pendiente.")
            return
        try:
            fila = guardar_gasto(gasto, username)
            await query.edit_message_text(formatear_guardado(fila), parse_mode="Markdown")
        except Exception as e:
            await query.edit_message_text(f"❌ Error al guardar: {str(e)[:100]}")

    elif data == "cancelar":
        gastos_pendientes.pop(chat_id, None)
        await query.edit_message_text("🚫 Gasto cancelado.")

    elif data.startswith("eliminar_"):
        gasto_id = int(data.split("_")[1])
        try:
            supabase.table("gastos").delete().eq("id", gasto_id).execute()
            await query.edit_message_text(f"🗑️ Gasto #{gasto_id} eliminado.")
        except Exception as e:
            await query.edit_message_text(f"❌ Error al eliminar: {str(e)[:100]}")

    elif data == "cancelar_eliminar":
        await query.edit_message_text("Ok, no se eliminó nada.")


async def manejar_audio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not usuario_autorizado(update):
        await update.message.reply_text("⛔ No estás autorizado.")
        return

    msg = await update.message.reply_text("🎙️ Transcribiendo audio...")
    try:
        audio       = update.message.voice or update.message.audio
        file        = await context.bot.get_file(audio.file_id)
        audio_bytes = await file.download_as_bytearray()
        texto       = await transcribir_audio(bytes(audio_bytes), "audio.ogg")
        logger.info(f"Transcripción: {texto}")
        await msg.delete()
        await procesar_y_preguntar(update, context, texto)
    except Exception as e:
        logger.error(f"Error audio: {e}", exc_info=True)
        await msg.edit_text(f"❌ Error: {str(e)[:100]}")


async def manejar_foto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Recibe una foto de boleta y extrae el gasto con GPT-4o vision."""
    if not usuario_autorizado(update):
        await update.message.reply_text("⛔ No estás autorizado.")
        return

    msg = await update.message.reply_text("📸 Analizando boleta...")
    username = update.effective_user.first_name or update.effective_user.username or "Usuario"

    try:
        # Tomar la foto de mayor resolución disponible
        foto    = update.message.photo[-1]
        file    = await context.bot.get_file(foto.file_id)
        img_bytes = await file.download_as_bytearray()

        await msg.edit_text("🤖 Extrayendo datos de la boleta...")

        gasto = await extraer_gasto_desde_imagen(bytes(img_bytes), username)
        gasto["_texto_original"] = f"[foto de boleta]"

        # Avisar si la confianza fue baja
        confianza = gasto.pop("confianza", "alta")
        if confianza == "baja":
            await update.message.reply_text(
                "⚠️ La boleta no era muy legible. Revisa el monto antes de confirmar."
            )

        gastos_pendientes[update.effective_chat.id] = gasto
        teclado = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Confirmar", callback_data="confirmar"),
            InlineKeyboardButton("❌ Cancelar",  callback_data="cancelar"),
        ]])
        await msg.edit_text(
            formatear_preview(gasto, username),
            parse_mode="Markdown",
            reply_markup=teclado,
        )

    except json.JSONDecodeError:
        await msg.edit_text("❌ No pude leer la boleta. Intenta con mejor iluminación o manda un audio.")
    except Exception as e:
        logger.error(f"Error foto: {e}", exc_info=True)
        await msg.edit_text(f"❌ Error al procesar la foto: {str(e)[:100]}")


async def manejar_texto(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not usuario_autorizado(update):
        return
    texto = update.message.text
    if texto.startswith("/"):
        return
    await procesar_y_preguntar(update, context, texto)


# ── main ──────────────────────────────────────────────────────────────────────

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Loguea errores de handlers, el Conflict se maneja en el loop principal."""
    if not isinstance(context.error, Conflict):
        logger.error(f"Error en handler: {context.error}", exc_info=context.error)


def construir_app():
    """Crea una instancia fresca de la aplicación con todos sus handlers."""
    app = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .request(HTTPXRequest(connect_timeout=10, read_timeout=30))
        .build()
    )
    app.add_handler(CommandHandler("start",    cmd_start))
    app.add_handler(CommandHandler("resumen",  cmd_resumen))
    app.add_handler(CommandHandler("eliminar", cmd_eliminar))
    app.add_handler(CallbackQueryHandler(manejar_callback))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, manejar_audio))
    app.add_handler(MessageHandler(filters.PHOTO, manejar_foto))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_texto))
    app.add_error_handler(error_handler)
    return app


async def run_bot():
    max_intentos = 10
    for intento in range(1, max_intentos + 1):
        # Nueva instancia en cada intento — Application no se puede reinicializar
        app = construir_app()
        try:
            logger.info(f"Intento {intento}/{max_intentos} de conectar...")
            await app.initialize()
            await app.bot.delete_webhook(drop_pending_updates=True)
            await app.start()
            await app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True,
            )
            logger.info("✅ Bot iniciado correctamente.")
            # Bloquea aquí hasta que Render detenga el proceso
            await asyncio.Event().wait()
        except Conflict:
            logger.warning(f"Conflicto detectado (intento {intento}/{max_intentos}). Esperando 10s...")
            try:
                await app.updater.stop()
                await app.stop()
                await app.shutdown()
            except Exception:
                pass
            if intento == max_intentos:
                logger.error("Se agotaron los reintentos. Saliendo.")
                raise
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Error inesperado: {e}", exc_info=True)
            try:
                await app.updater.stop()
                await app.stop()
                await app.shutdown()
            except Exception:
                pass
            raise


def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_bot())


if __name__ == "__main__":
    main()

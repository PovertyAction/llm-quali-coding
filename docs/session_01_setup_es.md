# Sesión 01 – Configuración local y primera llamada a una LLM API

## Objetivo de la sesión

Al final de esta sesión, deben poder:

- trabajar localmente en un proyecto de Python,
- entender qué significa usar un LLM a través de una API,
- configurar de forma segura una clave de API,
- ejecutar una primera solicitud exitosa a un Modelo de lenguaje de gran tamaño desde el computador,
- entender las respuestas de chat y el streaming,
- explorar la llamada a funciones (tools) con LLMs.

Esta sesión se enfoca en **infraestructura y fundamentos**.
No estamos construyendo una aplicación todavía — estamos asegurándonos de que la base funcione.

---

## Conceptos clave

### ¿Qué significa "usar un LLM"?

Los Modelos de lenguaje de gran tamaño no se ejecutan en el computador.
Se ejecutan en servidores remotos.

Cuando "usas un LLM" desde Python, estás:

- enviando una solicitud por internet,
- incluyendo entrada de texto (tu prompt),
- recibiendo una respuesta generada por el modelo.
s
Esta interacción se maneja a través de una **API (Interfaz de Programación de Aplicaciones)**.

**Punto clave:** No estás ejecutando el modelo—estás haciendo solicitudes HTTP a un servicio que ejecuta el modelo.

---

### La estructura del flujo de trabajo con LLM

La mayoría de los flujos de trabajo basados en LLM siguen la misma estructura de alto nivel:

```text
Entrada (texto, audio, datos)
↓
Llamada a la API del LLM
↓
Salida del modelo (texto, vectores, decisiones)
↓
Post-procesamiento y análisis
```

**En esta sesión nos enfocamos solo en la capa de llamada a la API del LLM.**
Todo lo demás se construye sobre esto.

---

### Claves de API y seguridad

Para acceder a una API de LLM, necesitas una clave de API.

**Reglas importantes:**

- Las claves de API te identifican a ti y a tu cuenta
- Las claves de API **nunca** deben codificarse directamente en los scripts
- Las claves de API **nunca** deben confirmarse en Git
- El uso de la API cuesta dinero—protege tus claves

**Cómo manejamos esto:**
En este proyecto, las claves de API se almacenan como variables de entorno.

Deberías tener un archivo llamado `.env` (no confirmado en Git) que contenga:

```bash
# Clave de API de Anthropic (recomendada)
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Clave de API de OpenAI (opcional - si quieres usar OpenAI)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

**Obtener una clave de API de Anthropic:**

1. Ve a [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
2. Regístrate o inicia sesión en tu cuenta
3. Crea una nueva clave de API
4. Copia la clave (comienza con `sk-ant-`)
5. Agrégala a tu archivo `.env` como se muestra arriba

**Obtener una clave de API de OpenAI:**

1. Ve a [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Regístrate o inicia sesión en tu cuenta
3. Crea una nueva clave de API (puede requerir agregar un método de pago)
4. Copia la clave (comienza con `sk-proj-`)
5. Agrégala a tu archivo `.env` como se muestra arriba

**Selección de proveedor:**

- Si solo `ANTHROPIC_API_KEY` está configurada → usa Anthropic (modelos Claude)
- Si solo `OPENAI_API_KEY` está configurada → usa OpenAI (modelos GPT)
- Si ambas están configuradas → usa Anthropic por defecto
- Configura `LLM_PROVIDER=openai` o `LLM_PROVIDER=anthropic` para elegir explícitamente

Tu código lee estas claves en tiempo de ejecución, sin exponerlas públicamente.

---

### Trabajar localmente (por qué esto importa)

En muchas demostraciones, los LLMs se muestran en notebooks o interfaces web.

Para trabajo real, usualmente necesitamos:

- **Reproducibilidad**: El mismo código produce los mismos resultados
- **Control de versiones**: Rastrear cambios a lo largo del tiempo
- **Dependencias explícitas**: Saber exactamente qué paquetes necesitas
- **Control sobre datos y PII**: Mantener la información sensible segura

Por eso estamos trabajando:

- localmente (en el computador),
- en un repositorio de Git,
- usando una estructura de proyecto de Python estándar.

---

## Actividad guiada

### Preparación: Familiarizarse el repositorio

Antes de continuar, asegúrate de haber clonado este repositorio y abierto en tu editor (VS Code o Positron).

**Carpetas clave:**

- `docs/` – materiales de aprendizaje y guías de sesiones
- `src/` – código Python reutilizable
- `examples/` – scripts que puedes ejecutar directamente
- `data/` – archivos de entrada pequeños, no sensibles

Para esta sesión, usaremos principalmente `examples/`.

---

### Paso 1 — Crear y activar el entorno (5 min)

Python depende mucho de paquetes externos.
Incluso tareas básicas a menudo requieren importar bibliotecas.

**En este proyecto:**

- las versiones de los paquetes están definidas en la configuración del proyecto,
- un entorno virtual aísla esos paquetes de otros proyectos.

**Ejecuta este comando:**

Si es la primera vez que configuras el proyecto, ejecuta:

```bash
just get-started
```

Si ya has configurado el proyecto antes, solo ejecuta:

```bash
just venv
```

**Este comando:**

- instala la versión correcta de Python,
- crea un entorno virtual,
- instala los paquetes requeridos.

**Después de eso, activa el entorno:**

| Shell      | Comando                                     |
|------------|---------------------------------------------|
| Bash       | `.venv/Scripts/activate`                    |
| PowerShell | `.venv/Scripts/activate.ps1`                |
| Nushell    | `overlay use .venv/Scripts/activate.nu`     |

**Qué observar:**
Ahora deberías ver tu terminal indicando un entorno activo (usualmente muestra `.venv` en el prompt).

---

### Paso 2 — Prueba tu conexión (5 min)

Antes de hacer algo complejo, siempre probamos la conexión.

**Ejecuta este script:**

```bash
python examples/basics/test_connection.py
```

**Este script hace exactamente una cosa:**

- Envía una solicitud mínima al LLM
- Imprime la respuesta

**Qué deberías ver:**

Si todo está configurado correctamente, deberías ver:

- Un mensaje de confirmación
- Una respuesta de texto corta del modelo

**Qué confirma esto:**

- ✅ Tu entorno funciona
- ✅ Tu clave de API está configurada correctamente
- ✅ Puedes comunicarte exitosamente con el LLM

**Si este paso falla, detente aquí y arréglalo antes de continuar.**
Todos los pasos posteriores dependen de que esto funcione.

---

## Llamadas de chat LLM

Ahora que tu conexión funciona, exploraremos diferentes patrones para usar la API de Chat Completions.

Estos scripts funcionan tanto con modelos de Anthropic (Claude) como de OpenAI (GPT), detectando automáticamente qué proveedor usar según tus claves de API. Están organizados en orden creciente de complejidad:

### Llamadas básicas de chat

**1. Chat simple (`chat.py`)**

```bash
python examples/basics/chat.py
```

**Qué hace:**

- Demuestra una llamada básica de chat
- Envía un solo mensaje
- Devuelve una respuesta completa

**Qué observar:**

- El patrón de solicitud-respuesta
- Cómo se estructuran los mensajes
- La salida completa del modelo

---

**2. Respuestas en streaming (`chat_stream.py`)**

```bash
python examples/basics/chat_stream.py
```

**Qué hace:**

- Agrega `stream=True` a la llamada de API
- Devuelve un generador que transmite la respuesta a medida que se genera
- Muestra tokens apareciendo uno a la vez (como la interfaz de ChatGPT)

**Qué observar:**

- La respuesta aparece progresivamente
- Mejor experiencia de usuario para respuestas largas
- Mismo resultado final, entrega diferente

**Cuándo usar streaming:**

- Construir interfaces de chat
- Generación de contenido largo
- Cuando quieres mostrar progreso a los usuarios

---

**3. Chat con historial (`chat_history.py`)**

```bash
python examples/basics/chat_history.py
```

**Qué hace:**

- Crea una interfaz de chat de ida y vuelta usando `input()`
- Rastrea mensajes pasados
- Envía el historial de conversación con cada llamada a la API

**Qué observar:**

- El modelo "recuerda" mensajes anteriores
- El contexto se acumula durante la conversación
- Cada llamada a la API incluye el historial completo

**Punto clave:** Los LLMs son sin estado—el modelo no recuerda nada. TÚ debes enviar el historial de conversación cada vez.

---

**4. Chat con historial y streaming (`chat_history_stream.py`)**

```bash
python examples/basics/chat_history_stream.py
```

**Qué hace:**

- Combina el historial de conversación con streaming
- Más similar a interfaces de chatbot de producción

**Qué observar:**

- Experiencia de chat completa con respuestas progresivas
- Cómo funciona la gestión del historial con streaming

---

## Llamada a funciones (Tools)

Estos scripts demuestran el uso de la función "tools" de la API de Chat Completions (también conocida como llamada a funciones).

**¿Qué es la llamada a funciones?**

En lugar de solo devolver texto, el modelo puede:

- Decidir cuándo llamar funciones definidas por el desarrollador
- Devolver argumentos estructurados que coincidan con el esquema de tu función
- Permitirte ejecutar código/APIs basándose en decisiones del modelo

**El flujo de trabajo:**

1. Declaras las funciones disponibles en el parámetro `tools`
2. El modelo puede responder con `message.tool_calls` en lugar de texto
3. Cada llamada a herramienta incluye el `name` de la función y los `arguments` JSON
4. Tu aplicación ejecuta la función y (opcionalmente) envía los resultados de vuelta

---

**1. Llamada a funciones básica (`function_calling_basic.py`)**

```bash
python examples/basics/function_calling_basic.py
```

**Qué hace:**

- Declara una sola función `lookup_weather`
- Pregunta al modelo con una pregunta relacionada con el clima
- Imprime la llamada a herramienta (si se detecta) o la respuesta normal
- NO ejecuta realmente la función

**Qué observar:**

- Cómo declarar esquemas de funciones
- Cuándo el modelo elige llamar vs. responder normalmente
- La estructura de las respuestas de llamadas a herramientas

**Ejemplo de salida:**

```text
Model chose to call: lookup_weather
Arguments: {"city_name": "Bogota"}
```

---

**2. Llamada a funciones con ejecución (`function_calling_call.py`)**

```bash
python examples/basics/function_calling_call.py
```

**Qué hace:**

- Extiende el ejemplo básico EJECUTANDO REALMENTE la función
- Declara el esquema de la función `lookup_weather`
- Cuando el modelo lo solicita, ejecuta la función con los argumentos proporcionados
- Devuelve datos meteorológicos simulados (18°C en Celsius)

**Qué observar:**

- Cómo analizar argumentos de llamadas a herramientas desde JSON
- Cómo ejecutar funciones basándose en decisiones del modelo
- El flujo de trabajo completo de llamada a funciones

**Ejemplo de salida:**

```text
Model chose to call: lookup_weather
Arguments: {"city_name": "Bogota"}

🌤️  Looking up weather for Bogota...
Function result: Currently 18°C and partly cloudy
```

**Punto clave:** El modelo decide CUÁNDO llamar, TU código decide CÓMO ejecutar.

---

**3. Traducción de documentos (`translate_ipa_document.py`)**

```bash
python examples/basics/translate_ipa_document.py
```

**Qué hace:**

- Lee el documento de IPA Best Bets en inglés (desde `data/`)
- Traduce todo el documento al español usando el LLM configurado
- Guarda la versión en español en `data/ipa-best-bets-2025-es.md`
- Preserva todo el formato markdown

**Qué observar:**

- Cómo manejar la traducción de documentos largos
- Operaciones de E/S de archivos con markdown
- Usar temperatura más baja (0.3) para traducción consistente
- Prompting de traducción profesional para contenido académico

**Cuándo usar llamada a funciones:**

- Cuando necesitas salidas estructuradas (JSON, no prosa)
- Cuando el modelo debe desencadenar acciones externas (llamadas a API, consultas a base de datos)
- Cuando quieres que el modelo use herramientas/plugins
- Para flujos de trabajo de múltiples pasos (el modelo decide qué hacer a continuación)

---

## Modelo mental: Anatomía de una llamada a la API de LLM

Cada llamada a la API de LLM tiene estos componentes:

**1. Cliente** — conexión autenticada

```python
client = get_client()  # Lee tu clave de API
```

**2. Selección de modelo** — qué motor usar

```python
# Proveedor autodetectado desde tus claves de API
provider = get_provider()  # Devuelve "anthropic" o "openai"

# Seleccionar modelo apropiado para el proveedor
model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5"
```

**3. Mensajes/prompt** — las instrucciones y datos

```python
messages=[
  {"role": "system", "content": "Eres un asistente útil"},
  {"role": "user", "content": "Hola"}
]
```

**4. Parámetros opcionales** — controlar comportamiento

```python
stream=True,  # Transmitir respuestas
tools=[...],  # Habilitar llamada a funciones
temperature=0.7,  # Controlar aleatoriedad
```

**5. Llamada a la API** — enviar la solicitud

```python
# Usando funciones adaptadoras para soporte multi-proveedor
response = create_completion(
    client=client,
    provider=provider,
    model=model,
    messages=messages,
    temperature=0.7
)
```

**6. Análisis de salida** — extraer lo que necesitas

```python
# Las funciones adaptadoras devuelven texto normalizado
text = create_completion(...)  # Devuelve string directamente

# Para llamadas a herramientas, usa el helper extract_tool_calls
tool_calls = extract_tool_calls(response, provider)
```

---

## Ejercicios rápidos (Después de la sesión de entrenamiento)

**Llamadas de chat:**

1. Modifica `chat.py` para hacer una pregunta diferente
2. Compara tiempos de respuesta entre `chat.py` y `chat_stream.py`
3. En `chat_history.py`, observa cómo el contexto afecta las respuestas

**Llamada a funciones:**

1. Compara `function_calling_basic.py` vs `function_calling_call.py` — ¿cuál es la diferencia?
2. Modifica la pregunta en `function_calling_call.py` para preguntar sobre una ciudad diferente
3. Agrega un nuevo parámetro a `lookup_weather` (ej., `forecast_days`)
4. Prueba qué sucede cuando haces una pregunta que no debería activar la función

---

## Problemas comunes y soluciones

### "No API keys found" o "ANTHROPIC_API_KEY not found"

→ Tu variable de entorno no está configurada o cargada
→ Verifica que el archivo `.env` existe y contiene tu clave de API
→ Asegúrate de haber activado el entorno virtual
→ Verifica el formato de la clave: las claves de Anthropic comienzan con `sk-ant-`, las de OpenAI con `sk-proj-`

### "ModuleNotFoundError: No module named 'anthropic'" o "'openai'"

→ Tu entorno no está activo o las dependencias no están instaladas
→ Ejecuta `just get-started` nuevamente
→ Asegúrate de ver `.venv` en tu prompt de terminal

### "¿Qué proveedor se está usando?"

→ Cada script imprime qué proveedor y modelo está usando
→ Busca salida como "Using anthropic with model: claude-haiku-4-5"
→ Configura la variable de entorno `LLM_PROVIDER` para elegir explícitamente

### "Streaming no funciona / sin salida"

→ Verifica que estás iterando sobre el stream correctamente
→ Busca el patrón `for chunk in response:`

### "La función nunca se llama"

→ Verifica que tu esquema de función coincida con lo que el modelo espera
→ Haz tu prompt más claro sobre cuándo usar la función
→ Prueba diferentes prompts que necesiten claramente la función

### "Los argumentos de la llamada a herramienta son incorrectos"

→ Verifica que la descripción de tu función sea clara
→ Verifica que las descripciones de parámetros expliquen lo que se espera
→ El modelo infiere de las descripciones—sé específico

---

## Lo que has aprendido

- ✅ Cómo configurar un entorno Python local para trabajo con LLM
- ✅ Cómo configurar claves de API de forma segura
- ✅ Qué es realmente una llamada a la API de LLM (solicitud HTTP a servicio remoto)
- ✅ Cómo hacer llamadas de chat básicas
- ✅ Cómo funciona el streaming y cuándo usarlo
- ✅ Cómo mantener el historial de conversación
- ✅ Qué es la llamada a funciones y cuándo es útil
- ✅ La estructura básica de cualquier interacción con LLM

---

## Patrones clave para recordar

**1. Llamada básica:** Pregunta simple → respuesta
**2. Streaming:** Entrega progresiva de respuesta
**3. Historial de conversación:** TÚ gestionas el contexto, el modelo es sin estado
**4. Llamada a funciones:** El modelo decide cuándo usar herramientas, TÚ las ejecutas

---

## Puente a las próximas sesiones

Ahora que tu base funciona, podemos construir sobre ella:

**Sesión 02 (siguiente):** Introducción a embeddings y RAG (`docs/session_02_embeddings_rag_es.md`)

- Aprende qué son los embeddings
- Comprende la búsqueda semántica
- Ve cómo trabajar con tus propios documentos

**Sesiones 03-04 (tarde):** Aplicaciones prácticas

- Codificación cualitativa con embeddings (`docs/session_03_quali_coding.md`)
- Construcción de un chatbot de conocimiento interno (`docs/session_04_rag_demo_es.md`)

**Punto clave:** La configuración que completaste hoy se reutiliza para todo lo demás. No necesitarás hacer esto nuevamente.

---

## Qué hicimos (y qué no hicimos)

### ✅ Hicimos

- Configurar un entorno Python local
- Entender cómo funcionan las APIs de LLM
- Hacer llamadas de chat básicas y en streaming
- Gestionar el historial de conversación
- Explorar la llamada a funciones (tools)

### ❌ No hicimos

- Ingeniería de prompts avanzada
- Evaluación de modelos
- Despliegue en producción
- Flujos de trabajo complejos de múltiples pasos
- Embeddings y RAG (eso es lo siguiente)

Piensa en esta sesión como **aprender los patrones básicos de API**.
Todo lo demás se construye sobre estos fundamentos.

---

## Soporte multi-proveedor

Todos los ejemplos de esta sesión soportan tanto modelos de Anthropic (Claude) como de OpenAI (GPT):

**Cómo funciona:**

- Los scripts detectan automáticamente qué clave de API has configurado
- Las diferencias específicas del proveedor (formato de API, parámetros) se manejan de forma transparente
- Puedes cambiar de proveedor cambiando qué clave de API está configurada en `.env`

**Modelos soportados:**

- **Anthropic:** claude-haiku-4-5 (usado en todos los ejemplos)
- **OpenAI:** gpt-4o-mini (rápido, rentable) y gpt-4o (mayor calidad)

**¿Por qué multi-proveedor?**

- Comparar diferentes modelos para tu caso de uso
- Evitar dependencia de un proveedor
- Aprender patrones de API que funcionan entre proveedores
- Educativo: Ver cómo funcionan diferentes APIs de LLM

---

## Agradecimientos

Algunos ejercicios y ejemplos en esta sesión fueron adaptados o inspirados por el repositorio [Azure Python OpenAI Samples](https://github.com/Azure-Samples/python-openai-demos).

Los ejemplos soportan tanto APIs de Anthropic como de OpenAI con detección automática de proveedor.

---

**¿Listo para más?** Pasa a la Sesión 02 para aprender sobre embeddings y RAG.

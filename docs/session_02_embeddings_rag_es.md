# Sesión 02 — Embeddings y RAG

## Objetivo de la sesión

Al final de esta sesión, deberían ser capaz de:

- explicar qué es un **embedding** y para qué se utiliza,
- generar embeddings para textos,
- medir similitud (coseno / producto punto) y recuperar los *top-k* fragmentos,
- construir un **mini-RAG**: *primero buscar, luego responder*.

> Esta sesión es **fundamental**: lo que aprendas se aplica a muchos casos de uso (chatbots internos, búsqueda semántica, clasificación asistida, etc.).

---

## Conceptos clave

### ¿Qué es un embedding?

Un embedding es una **representación numérica** del significado de un texto.

- Es un **vector** (lista de números)
- Vive en un **espacio vectorial**
- Los textos con significado similar tienden a estar **cerca** en ese espacio

**Regla importante:** Solo puedes comparar embeddings si fueron generados con el **mismo modelo** (mismo espacio).

**Ejemplo:**

```text
Texto: "La encuesta fue realizada en aldeas rurales"
↓
Embedding: [0.023, -0.145, 0.892, ..., 0.334]
(1536 números con text-embedding-3-small)
```

Textos similares obtienen vectores similares. Significados diferentes obtienen vectores distantes.

---

### Similitud: similitud coseno y producto punto

La comparación más común es la **similitud coseno** (intuición: "ángulo entre vectores").

Muchos modelos de embeddings producen vectores **normalizados**, y entonces:

- la similitud coseno se simplifica a un **producto punto**
- esto es eficiente y escala bien

**En la práctica:**

- Puntuación cercana a 1 = significado muy similar
- Puntuación cercana a 0 = no relacionado
- Puntuación cercana a -1 = significado opuesto (raro)

---

### ¿Qué es RAG?

RAG = Generación Aumentada por Recuperación (Retrieval-Augmented Generation)

En lugar de pedirle al LLM que "conozca" tus documentos, tú:

1. **Recuperación**: Encuentra fragmentos relevantes (embeddings + búsqueda)
2. **Aumentación**: Agrega esos fragmentos al prompt
3. **Generación**: El LLM responde usando ese contexto

Frase útil:
> **Primero buscar, luego responder.**

**Idea clave:** RAG no es fine-tuning. No estás cambiando el modelo—estás cambiando el **contexto**.

**Por qué RAG es importante:**

- Funciona con datos privados/propietarios
- No necesita reentrenar modelos
- Fácil de actualizar la base de conocimiento
- Puede citar fuentes
- Rentable a escala

---

## El flujo de trabajo de RAG

```text
Tus documentos
↓
Dividir en fragmentos
↓
Generar embeddings (una vez)
↓
Almacenar embeddings
---
Pregunta del usuario
↓
Generar embedding de la pregunta
↓
Buscar fragmentos similares (recuperación)
↓
Construir prompt con fragmentos (aumentación)
↓
LLM genera respuesta (generación)
```

---

## Actividad guiada

### Paso 0 — Crear y activar el entorno

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

### Preparación

Asegúrate de tener:

- entorno activo (`.venv`)
- `.env` con `OPENAI_API_KEY`
- el repositorio abierto en VS Code

---

### Paso A — Generar embeddings (10 min)

Ejecuta:

```bash
python examples/embeddings/01_generate_embeddings.py
```

**Lo que deberías ver:**

- Impresión del tamaño del vector (ej., 1536 dimensiones)
- Vista previa de los primeros valores
- Confirmación de que la API funciona para embeddings

**Lo que está sucediendo:**

- Tu texto se está enviando a la API de embeddings de OpenAI
- La API devuelve una representación vectorial
- Este vector captura el significado semántico de tu texto

**Observación clave:** Nota cómo diferentes textos producen diferentes vectores, pero la longitud del vector siempre es la misma (1536 para `text-embedding-3-small`).

---

### Paso B — Búsqueda semántica simple (15 min)

Ejecuta:

```bash
python examples/embeddings/02_similarity_search.py
```

**Este script:**

1. Lee varios documentos cortos
2. Los divide en fragmentos (fragmentación simple)
3. Genera embeddings por fragmento
4. Genera embedding de una pregunta
5. Calcula similitudes y muestra los top-k

**Qué observar:**

- Los fragmentos recuperados deberían "tener sentido" con tu pregunta
- El top-1 y top-3 deberían ser razonables
- Nota las puntuaciones de similitud (mayor = más relevante)

**Lo que está sucediendo:**

- Cada fragmento se convierte en un vector
- Tu pregunta también se convierte en un vector
- Calculamos qué tan cerca está cada vector de fragmento del vector de la pregunta
- Clasificamos por similitud y devolvemos las mejores coincidencias

**Idea clave:** Esta es búsqueda semántica—estamos encontrando coincidencias de significado, no coincidencias de palabras clave.

---

### Paso C — Mini-RAG (10–15 min)

Ejecuta:

```bash
python examples/embeddings/03_mini_rag.py
```

**Este script hace lo mismo que el Paso B, pero adicionalmente:**

- Construye un prompt que incluye los fragmentos recuperados
- Le pide al LLM que responda usando **solo ese contexto**
- Imprime la respuesta

**Qué observar:**

- La respuesta debería hacer referencia al contenido recuperado
- Si la pregunta no está cubierta por el contexto, el modelo debería decirlo (según la instrucción)
- Compara la respuesta con los fragmentos que fueron recuperados

**Lo que está sucediendo:**

```text
1. Usuario pregunta: "¿Qué método de muestreo se utilizó?"
2. Sistema recupera fragmentos relevantes sobre muestreo
3. Sistema construye prompt: "Basado en estos documentos: [fragmentos], responde: [pregunta]"
4. LLM genera respuesta fundamentada en esos fragmentos
```

**Idea clave:** El LLM no "conoce" tus documentos—tú le estás dando las partes relevantes en el momento de la consulta.

---

## Modelo mental: RAG vs. otros enfoques

| Enfoque | Cuándo usar | Pros | Contras |
|----------|-------------|------|------|
| **Llamada simple al LLM** | Tareas generales, no se necesita conocimiento específico | Simple, rápido | No puede acceder a tus datos |
| **Contexto largo** | Base de conocimiento pequeña y estática | No se necesita recuperación | Costoso, puede perder detalles |
| **RAG** | Base de conocimiento grande y cambiante | Eficiente, escalable, citable | Requiere configuración de embeddings |
| **Fine-tuning** | Necesitas que el modelo "hable" tu dominio | El modelo aprende estilo/dominio | Costoso, difícil de actualizar |

Para la mayoría de casos de uso en investigación → RAG

---

## Ejercicios rápidos (extra)

1. **Cambiar profundidad de recuperación:** Modifica `TOP_K` de 3 a 5 y compara resultados. ¿Ayuda más contexto?

2. **Experimentar con fragmentación:** Cambia la estrategia de fragmentación (ej., por párrafos vs. tamaño fijo) y observa cómo cambian los top-k.

3. **Probar casos extremos:** Escribe una pregunta "difícil" (poca evidencia en los documentos) y observa cómo se comporta el mini-RAG. ¿Alucina o admite que no sabe?

4. **Comparar con palabras clave:** Intenta encontrar la misma información usando búsqueda por palabras clave (Ctrl+F). Nota la diferencia con la búsqueda semántica.

---

## Actividad final — Construye tu propio mini-RAG

Ahora que has visto cómo funciona el ejemplo de mini-RAG, ¡es hora de construir el tuyo!

### Instrucciones

**Objetivo:** Crear un sistema mini-RAG para responder preguntas sobre procedimientos de adquisiciones.

**Conjunto de datos:** Usa el archivo ubicado en `data/exercise/How to Create a Procurement Request.md`

**Tu tarea:**

1. **Leer y fragmentar** el documento de adquisiciones
2. **Generar embeddings** para cada fragmento
3. **Implementar el flujo de trabajo RAG**:
4.
   - Aceptar preguntas del usuario
   - Recuperar fragmentos relevantes usando búsqueda por similitud
   - Construir un prompt con el contexto
   - Generar respuestas usando el LLM

5. **Probar con preguntas** como:
6.
   - "¿Cuál es el tiempo mínimo de anticipación requerido para la fecha de entrega según el nivel de solicitud de adquisición en el sistema ProcessMaker de IPA, y qué sucede si un usuario no ha visitado ProcessMaker recientemente?"
   - "Si un usuario de IPA no encuentra el Código de Asignación de Subvención que necesita al crear una solicitud de adquisición en ProcessMaker, ¿a quién debe contactar y en qué se basa la lista de códigos disponibles?"

**Punto de partida:** Usa `examples/03_mini_rag.py` como tu código de referencia y adáptalo para trabajar con el documento de adquisiciones.

**Puedes:**

- Usar Claude o GitHub Copilot Chat para ayudarte a escribir el código
- Hacer preguntas sobre la implementación
- Experimentar con diferentes estrategias de fragmentación
- Ajustar el número de fragmentos recuperados (`TOP_K`)

**Resultado esperado:** Un script funcional que pueda responder preguntas relacionadas con adquisiciones basándose en el contenido del documento.

---

## Problemas comunes y soluciones

**"Los embeddings son todos similares (puntuaciones cercanas a 1)"**
→ Tus documentos podrían ser demasiado similares, o los fragmentos demasiado pequeños. Prueba con contenido más diverso.

**"Los resultados principales no tienen sentido"**
→ Revisa tu estrategia de fragmentación. ¿Los fragmentos preservan coherencia semántica?

**"El LLM ignora el contexto"**
→ Haz tu instrucción del prompt más clara: "Responde SOLO usando el contexto proporcionado."

**"Demasiado lento"**
→ Los embeddings se generan una vez y se almacenan. Solo el embedding de la consulta sucede en tiempo real.

---

## Lo que has aprendido

- ✅ Los embeddings convierten texto en vectores buscables
- ✅ La búsqueda por similitud encuentra contenido semánticamente relacionado
- ✅ Patrón RAG: recuperar → aumentar → generar
- ✅ Esto funciona sin fine-tuning o reentrenamiento de modelos

---

## Puente a las próximas sesiones

Lo que viene después:

- **Sesión 03** (esta tarde): Aplicar esto a flujos de trabajo de codificación cualitativa
- **Sesión 04** (esta tarde): Construir un chatbot de conocimiento completo con tus documentos internos

Consideraciones futuras:

- Fragmentación más robusta (tokens, solapamiento)
- Almacenamiento eficiente de embeddings (archivos / base de datos vectorial)
- Trazabilidad (citas, IDs de fragmentos)
- Evaluación: ¿Estamos recuperando el contenido correcto? ¿La respuesta usa el contexto correctamente?

Esta sesión te da la base técnica para construir estas aplicaciones.

---

## Conclusiones clave

1. **Embeddings = búsqueda semántica**: Encuentra por significado, no por palabras clave
2. **RAG = generación fundamentada**: El LLM usa TUS datos, no solo conocimiento de entrenamiento
3. **El patrón es reutilizable**: El mismo enfoque funciona para muchas aplicaciones de investigación
4. **No se necesita entrenar el modelo**: Esto es ingeniería de prompts + recuperación inteligente

---

## Recursos

- Guía de Embeddings de OpenAI: <https://platform.openai.com/docs/guides/embeddings>
- Entendiendo la similitud coseno: <https://www.pinecone.io/learn/vector-similarity/>
- Mejores prácticas de RAG: <https://www.anthropic.com/index/contextual-retrieval>

---

**¿Listo para más?** Continúa con la Sesión 03 esta tarde para ver los embeddings en acción para investigación cualitativa.

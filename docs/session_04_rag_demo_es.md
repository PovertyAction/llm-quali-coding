# Sesión 04 — RAG: Generación Aumentada por Recuperación para proyectos

## Objetivo de la sesión

Al final de esta sesión, deberías poder:

- convertir **PDFs a markdown** para procesamiento de texto,
- crear **embeddings** de documentación de proyectos,
- construir una **base de datos vectorial** para búsqueda semántica,
- implementar un sistema **RAG (Generación Aumentada por Recuperación)**,
- consultar conocimiento de proyectos (tasas de respuesta, prácticas de campo, desafíos),
- entender cuándo RAG mejora las respuestas del LLM vs. prompting directo.

> Esta sesión se basa en las Sesiones 02 y 03, aplicando embeddings a documentación de proyectos y gestión de conocimiento institucional.
>
> **Importante:** Este es un flujo de trabajo educativo. En proyectos reales, considera la privacidad de datos y sensibilidad de documentos al construir bases de conocimiento.

---

## Conceptos clave

### ¿Qué es RAG?

**RAG (Retrieval Augmented Generation / Generación Aumentada por Recuperación)** combina dos pasos:

1. **Recuperación**: Encuentra documentos relevantes de tu base de conocimiento
2. **Generación**: El LLM genera respuestas usando el contexto recuperado

**¿Por qué RAG?**

- Los LLMs no conocen los detalles específicos de tus proyectos
- Entrenar modelos personalizados es costoso y consume tiempo
- RAG da a los LLMs acceso a tus documentos sin reentrenamiento
- Mantiene información actualizada conforme los documentos cambian
- Provee citaciones de fuentes para las respuestas

**Concepto clave:** RAG convierte tus documentos de proyecto en una base de conocimiento consultable que el LLM puede referenciar al responder preguntas.

---

### El flujo de trabajo RAG

```text
PDFs de Proyectos (3 reportes)
↓
Convertir PDFs → Markdown
↓
Dividir documentos en chunks (por sección, párrafo o unidad semántica)
↓
Generar embeddings para cada chunk
↓
Almacenar en base de datos vectorial (CSV o vector DB)
↓
Usuario hace una pregunta
↓
Embedizar la pregunta
↓
Recuperar top-k chunks más similares
↓
Enviar pregunta + chunks recuperados al LLM
↓
LLM genera respuesta con contexto
↓
Retornar respuesta con citaciones de fuentes
```

---

### Conversión de PDF a markdown

**Problema:** Los PDFs no son fácilmente procesados por LLMs. Contienen formato, imágenes, tablas y metadatos que complican la extracción de texto.

**Solución:** Convertir PDFs a texto markdown limpio que preserve la estructura (encabezados, listas, párrafos) pero elimine formato innecesario.

**Beneficios:**

- Texto limpio y parseable para chunking
- Preserva estructura del documento (secciones, subsecciones)
- Más fácil de procesar programáticamente
- Mejor calidad de embeddings

**Herramientas:** Usaremos `pymupdf4llm` que extrae texto inteligentemente mientras preserva la estructura markdown.

---

### Estrategias de chunking para documentos de proyectos

**Objetivo:** Dividir documentos en unidades significativas que puedan ser embedizadas y recuperadas independientemente.

**Estrategias comunes:**

| Estrategia | Mejor para | Tamaño de chunk |
|----------|----------|------------|
| **Por sección** | Reportes estructurados con encabezados claros | Variable (100-1000 tokens) |
| **Tamaño fijo** | Documentos homogéneos | 500-1000 tokens con superposición |
| **Unidades semánticas** | Contenido mixto (párrafos, listas, tablas) | Variable, consciente del contexto |
| **Por metadatos** | Corpus multi-documento | Depende de la estructura del documento |

**Para reportes de proyectos, recomendamos:**

- **Chunking basado en secciones**: Cada sección principal se convierte en un chunk
- **Preservar contexto**: Incluir encabezados de sección en los chunks
- **Tamaño óptimo**: 300-800 tokens por chunk (aproximadamente 1-3 párrafos)
- **Agregar superposición**: Incluir 50-100 tokens del chunk anterior para continuidad

---

### Recuperación: Encontrando chunks relevantes

**Cómo funciona:**

1. Usuario pregunta: "¿Cuáles fueron los principales desafíos de campo en el Proyecto A?"
2. Sistema embediza la pregunta → vector
3. Compara el vector de la pregunta con todos los vectores de chunks (similitud coseno)
4. Retorna top-k chunks más similares (típicamente k=3-5)
5. Estos chunks proveen contexto al LLM

**Estrategias de recuperación:**

| Estrategia | Cuándo usar |
|----------|-------------|
| **Recuperación Top-k** | Más común, simple y efectivo |
| **Basado en umbral** | Cuando quieres puntaje mínimo de relevancia |
| **Híbrido (palabra clave + semántico)** | Para documentos técnicos con términos específicos |
| **Filtrado por metadatos** | Al consultar proyectos o secciones específicas |

**Concepto clave:** Una buena recuperación es crítica—si no se recuperan los chunks correctos, el LLM no puede generar respuestas precisas.

---

### Generación: LLM con contexto

**Estructura del prompt:**

```text
Eres un asistente de investigación ayudando a analizar documentación de proyectos.

Contexto de documentos de proyectos:
---
[Chunk 1: Sección más relevante...]
[Chunk 2: Siguiente sección relevante...]
[Chunk 3: Contexto adicional...]
---

Pregunta del usuario: ¿Cuáles fueron los principales desafíos de campo en el Proyecto A?

Instrucciones:
- Responde basándote SOLO en el contexto proporcionado
- Cita documentos y secciones específicas
- Si la información no está en el contexto, dilo
- Sé específico y conciso
```

**Por qué funciona:**

- El LLM tiene información específica del proyecto en contexto
- Respuestas fundamentadas (no alucinadas)
- Puede citar fuentes
- Respeta el contenido del documento

---

### Cuándo usar RAG vs. prompting directo

| Enfoque | Mejor para | Ventajas | Desventajas |
|----------|----------|------|------|
| **Prompting directo** | Preguntas generales, hechos conocidos | Simple, rápido, económico | Sin conocimiento específico del proyecto |
| **RAG** | Consultas específicas de proyectos | Preciso, citado, actualizable | Requiere configuración, más lento |
| **Fine-tuning** | Patrones repetidos, tareas especializadas | Mejor rendimiento | Costoso, conocimiento estático |

**Usa RAG cuando:**

- Consultas información específica de proyectos (prácticas de campo, tasas de respuesta, lecciones aprendidas)
- Necesitas citaciones de fuentes
- Los documentos se actualizan frecuentemente
- Quieres combinar conocimiento de múltiples proyectos

**Usa prompting directo cuando:**

- Haces preguntas generales sobre métodos de investigación
- Lluvia de ideas o ideación
- No se necesita referencia a documentos

---

## Actividades guiadas

### Preparación

Asegúrate de tener:

- Crear entorno (`just venv`)
- `.env` con `OPENAI_API_KEY`
- El repo abierto en VS Code
- Completadas las Sesiones 02 y 03 (embeddings y codificación cualitativa)
- Tres reportes PDF de proyectos en `data/project_reports/`
- Activar entorno: `.venv/Scripts/activate.ps1`

Todos los paquetes requeridos (pymupdf4llm, openai, pandas, numpy, scikit-learn, etc.) se instalarán automáticamente cuando ejecutes `just venv`.

---

### Actividad A — Convertir PDFs a markdown

Ejecutar:

```bash
python examples/rag/01_pdf_to_markdown.py
```

**Lo que hace este script:**

1. Escanea `data/project_reports/` buscando archivos PDF
2. Para cada PDF:
   - Extrae texto usando `pymupdf4llm`
   - Preserva estructura markdown (encabezados, listas, párrafos)
   - Limpia artefactos de formato
   - Guarda en `outputs/markdown/nombre_proyecto.md`
3. Genera un reporte resumen mostrando:
   - Número de PDFs procesados
   - Tamaños de documentos
   - Estructura de secciones detectada

**Qué observar:**

- ¿Cuántas secciones se detectaron por documento?
- ¿Está limpia y legible la estructura markdown?
- ¿Las tablas y listas se preservaron correctamente?
- ¿Hay errores de extracción o artefactos?

**Problemas comunes:**

- **PDFs escaneados (imágenes)**: No se extraerán bien—necesitan OCR primero
- **Tablas complejas**: Pueden no convertirse perfectamente—inspeccionar salida
- **Layouts multi-columna**: El orden del texto podría estar mal—verificar manualmente

**Qué está pasando:**

- El script identifica inteligentemente la estructura del documento
- Los encabezados (H1, H2, H3) se detectan por tamaño y formato de fuente
- Los párrafos se preservan como bloques de texto
- Las listas y viñetas se mantienen
- Esto crea texto limpio y listo para LLM

**Concepto clave:** Una buena extracción de markdown es la base de RAG de calidad. Basura entra, basura sale—dedica tiempo a validar tus documentos convertidos.

---

### Actividad B — Crear chunks de documentos con embeddings

Ejecutar:

```bash
python examples/rag/02_create_document_chunks.py
```

**Lo que hace este script:**

1. Carga todos los archivos markdown de `outputs/markdown/`
2. **Parsea estructura del documento**: Identifica secciones, subsecciones, párrafos
3. **Divide por unidades semánticas**:
   - Cada sección principal se convierte en un chunk
   - Incluye encabezado de sección para contexto
   - Tamaño objetivo: 300-800 tokens
   - Superpone 100 tokens entre chunks para continuidad
4. **Agrega metadatos** a cada chunk:
   - Documento fuente (nombre del proyecto)
   - Título de sección
   - Referencia de página/ubicación
   - Índice de chunk
5. Genera embeddings para cada chunk
6. Guarda en `outputs/01_document_chunks.csv`

**Estrategia de chunking:**

```text
Proyecto_A.md
├─ Introducción (200 tokens) → Chunk 1 [metadata: Proyecto_A, Introducción]
├─ Metodología
│  ├─ Diseño Muestral (450 tokens) → Chunk 2 [metadata: Proyecto_A, Metodología > Muestreo]
│  └─ Recolección Datos (600 tokens) → Chunk 3 [metadata: Proyecto_A, Metodología > Recolección]
├─ Hallazgos
│  ├─ Tasas Respuesta (350 tokens) → Chunk 4 [metadata: Proyecto_A, Hallazgos > Tasas]
│  └─ Desafíos Campo (700 tokens) → Chunk 5 [metadata: Proyecto_A, Hallazgos > Desafíos]
└─ Conclusiones (400 tokens) → Chunk 6 [metadata: Proyecto_A, Conclusiones]
```

**Qué observar:**

- ¿Cuántos chunks se crearon por documento?
- ¿Cuál es el tamaño promedio de chunk (tokens)?
- ¿Los chunks son muy pequeños o muy grandes?
- ¿Los metadatos se ven correctos (nombres de proyectos, secciones)?
- ¿Se generaron embeddings para todos los chunks?

**Estructura CSV de salida:**

```text
chunk_id | project | section | text | embedding | token_count
---------|---------|---------|------|-----------|------------
chunk_1  | Proyecto_A | Introducción | [texto...] | [0.12, -0.45, ...] | 215
chunk_2  | Proyecto_A | Metodología > Muestreo | [texto...] | [0.34, 0.21, ...] | 430
...
```

**Qué está pasando:**

- Cada chunk se convierte en una unidad buscable en tu base de conocimiento
- Los embeddings capturan el significado semántico de cada sección
- Los metadatos permiten filtrado (ej., "solo buscar Proyecto A")
- Esto crea una base de datos vectorial de conocimiento del proyecto

**Concepto clave:** ¡Las decisiones de chunking importan! Los chunks muy pequeños carecen de contexto; los muy grandes diluyen la relevancia. El chunking basado en secciones balancea ambos.

---

### Actividad C — Construir base de datos vectorial y probar recuperación

Ejecutar:

```bash
python examples/rag/03_build_vector_database.py
```

**Lo que hace este script:**

1. Carga chunks de documentos con embeddings de la Actividad B
2. **Indexa chunks** para búsqueda rápida de similitud:
   - Convierte embeddings a arrays numpy
   - Construye índice de búsqueda de similitud coseno
   - Almacena metadatos para filtrado
3. **Prueba recuperación** con consultas de ejemplo:
   - "¿Cuáles fueron las tasas de respuesta en cada proyecto?"
   - "¿Qué desafíos de campo se encontraron?"
   - "¿Cuáles son las mejores prácticas para participación comunitaria?"
4. Para cada consulta:
   - Embediza la pregunta
   - Recupera top-5 chunks más similares
   - Muestra puntajes de similitud
   - Muestra metadatos de fuente (proyecto, sección)
5. Guarda base de datos indexada en `outputs/02_vector_database.pkl`

**Salida de ejemplo:**

```text
Consulta: "¿Cuáles fueron las tasas de respuesta en cada proyecto?"

Top 5 chunks recuperados:
1. [Puntaje: 0.87] Proyecto_A > Hallazgos > Tasas de Respuesta
   "La tasa de respuesta final fue 68%, superando nuestra meta del 65%..."
   
2. [Puntaje: 0.82] Proyecto_B > Resultados > Métricas de Participación
   "Alcanzamos una tasa de finalización del 72% entre participantes inscritos..."
   
3. [Puntaje: 0.79] Proyecto_C > Calidad de Datos > Análisis de Cobertura
   "La tasa inicial fue 55%, con seguimiento aumentando a 63%..."
```

**Qué observar:**

- ¿Los chunks recuperados son realmente relevantes a la consulta?
- ¿Cuáles son los puntajes de similitud? (>0.7 es bueno, >0.8 es excelente)
- ¿Se recuperan chunks de múltiples proyectos?
- ¿Hay chunks irrelevantes en el top-5?
- ¿La recuperación captura respuestas directas y contexto relacionado?

**Qué está pasando:**

- El sistema hace búsqueda semántica—encuentra coincidencias de significado, no de palabras clave
- La consulta "tasas de respuesta" recupera chunks sobre participación, finalización, cobertura—todo semánticamente relacionado
- Los metadatos muestran de qué proyecto y sección proviene cada chunk
- Esta es la parte de **recuperación** de RAG

**Concepto clave:** Prueba la recuperación independientemente antes de agregar el LLM. Si la recuperación falla, todo el sistema RAG falla.

**Prueba esto:** Modifica las consultas de prueba y ve qué se recupera. Prueba preguntas específicas vs. generales.

---

### Actividad D — Implementar sistema RAG (básico)

Ejecutar:

```bash
python examples/rag/04_rag_system_basic.py
```

**Lo que hace este script:**

1. Carga la base de datos vectorial de la Actividad C
2. **Interfaz interactiva de consultas**:
   - Solicita al usuario una pregunta
   - Recupera top-k chunks relevantes (k=5 por defecto)
   - Construye un prompt con contexto recuperado
   - Envía al LLM para generación de respuesta
   - Retorna respuesta con citaciones de fuente
3. **Implementa pipeline RAG**:
   - Consulta → Embedizar → Recuperar → Generar → Citar
4. Guarda historial de conversación en `outputs/09_rag_conversations.jsonl`

**Plantilla de prompt RAG:**

```text
Eres un asistente de investigación ayudando a analizar lecciones aprendidas de tres proyectos de salud comunitaria.

Contexto de documentación de proyectos:
---
Documento: Proyecto_A, Sección: Hallazgos > Tasas de Respuesta
Contenido: [chunk recuperado 1...]

Documento: Proyecto_B, Sección: Metodología > Prácticas de Campo
Contenido: [chunk recuperado 2...]

Documento: Proyecto_C, Sección: Desafíos > Participación Comunitaria
Contenido: [chunk recuperado 3...]
---

Pregunta del usuario: {pregunta_usuario}

Instrucciones:
- Responde basándote SOLO en el contexto proporcionado de documentos de proyectos
- Cita proyectos y secciones específicos al hacer afirmaciones
- Si una pregunta no puede responderse del contexto, di: "Esta información no está disponible en los documentos de proyectos proporcionados"
- Sé específico y proporciona detalles de los documentos
- Compara entre proyectos cuando sea relevante
```

**Interacción de ejemplo:**

```text
Tú: ¿Cuáles fueron los principales desafíos con la participación comunitaria?

Sistema RAG:
[Recuperando chunks relevantes de 3 proyectos...]
[Recuperados 5 chunks con similitud promedio: 0.81]

Basado en la documentación del proyecto:

**Proyecto A** enfrentó desafíos con construcción de confianza en acercamiento inicial. 
El reporte nota: "Los miembros de la comunidad fueron inicialmente escépticos de investigadores 
externos, requiriendo 3-4 reuniones preliminares antes del consentimiento" (Proyecto_A > 
Hallazgos > Desafíos).

**Proyecto B** encontró barreras de idioma. Reportan: "Falta de materiales 
en lenguas indígenas limitó participación entre el 30% de la población objetivo" 
(Proyecto_B > Lecciones Aprendidas > Accesibilidad).

**Proyecto C** identificó problemas de tiempo: "Conflictos de horario con temporadas 
agrícolas resultaron en menor asistencia a reuniones comunitarias" (Proyecto_C > Notas 
de Campo > Programación).

Los tres proyectos recomiendan: asociación comunitaria temprana, materiales 
culturalmente apropiados y programación flexible.
```

**Qué observar:**

- ¿La respuesta se mantiene fundamentada en los documentos?
- ¿Las citaciones son específicas y precisas?
- ¿El sistema compara entre proyectos cuando se le pregunta?
- ¿Qué pasa cuando preguntas sobre información no en los documentos?
- ¿La calidad de respuesta es mejor que sin RAG?

**Qué está pasando:**

- **Recuperación**: Encuentra chunks semánticamente más similares a tu pregunta
- **Aumentación**: Agrega estos chunks como contexto en el prompt
- **Generación**: El LLM lee el contexto y genera respuesta informada
- **Citación**: El LLM referencia proyectos/secciones específicos (si se instruye)

**Concepto clave:** RAG da a los LLMs "memoria de trabajo" de tus documentos. El LLM no memoriza tus reportes—lee secciones relevantes al responder.

**Prueba preguntando:**

- "¿Qué tasas de respuesta logró cada proyecto?"
- "¿Cuáles son las mejores prácticas para recolección de datos de campo entre estos proyectos?"
- "¿Qué desafíos fueron únicos del Proyecto B?"
- "¿Qué proyecto tuvo la mayor satisfacción de participantes?"
- "¿Qué recomendarías para un proyecto similar basado en estas lecciones?"

---

### Actividad E — RAG con filtrado de metadatos (avanzado)

Ejecutar:

```bash
python examples/rag/05_rag_with_metadata.py
```

**Lo que hace este script:**

1. Extiende el sistema RAG básico con **filtrado de metadatos**:
   - Filtrar por proyecto específico: "Solo buscar Proyecto A"
   - Filtrar por tipo de sección: "Solo secciones de metodología"
   - Filtrar por tipo de documento: "Solo hallazgos y conclusiones"
2. **Interfaz de consulta mejorada**:
   - Permite a usuarios especificar filtros antes de hacer preguntas
   - Muestra cuántos chunks coinciden con filtros antes de recuperación
   - Muestra resultados de recuperación filtrados
3. **Filtrado inteligente**:
   - Detección de filtro en lenguaje natural: "¿Qué dijo el Proyecto B sobre tasas de respuesta?"
   - Extracción automática de proyecto de la consulta
   - Búsquedas específicas de sección

**Ejemplo con filtrado:**

```text
Tú: [project:Conectividad] ¿Cuáles fueron las principales lecciones?

Sistema RAG:
[Aplicando filtro: project = Conectividad]
[Chunks filtrados: 14 → 11 restantes]
[Recuperando solo de Conectividad...]

Basado en documentación de Conectividad:

Las principales lecciones fueron:
1. [Hallazgos específicos del proyecto Conectividad...]
2. [Ideas clave...]
...
```

**Cómo usar los filtros - Ejemplos prácticos:**

| Tipo de pregunta | Sintaxis del filtro |
|------------------|---------------------|
| **Sin filtro** (busca en todos los proyectos) | `¿Cuáles fueron los principales desafíos?` |
| **Filtrar por proyecto específico** | `[project:Conectividad] ¿Qué lecciones se aprendieron?` |
| **Filtrar por otro proyecto** | `[project:Sisben] ¿Cuáles fueron los resultados?` |
| **Filtrar por sección** | `[section:Metodología] ¿Qué metodología se usó?` |
| **Combinar filtros** | `[project:Conectividad][section:Resultados] ¿Cuáles fueron los resultados?` |
| **Comparar proyectos** (sin filtro) | `¿Cuáles son las diferencias entre los proyectos?` |

**Ejemplos para tus proyectos:**

```text
# Pregunta específica sobre Conectividad
Your question: [project:Conectividad] ¿Qué desafíos enfrentó este proyecto?

# Pregunta específica sobre Sisben
Your question: [project:Sisben] ¿Cuáles fueron las lecciones aprendidas?

# Comparar ambos proyectos (sin filtro)
Your question: ¿Cuáles son las diferencias clave entre los proyectos?

# Usar coincidencia parcial
Your question: [project:Conect] ¿Qué se aprendió?
(funciona igual que [project:Conectividad])
```

**Nota:** Los filtros usan coincidencia parcial (`contains`), así que puedes escribir solo parte del nombre del proyecto.

**Qué observar:**

- ¿El filtrado mejora la relevancia de respuestas?
- ¿Cómo cambia el conteo de chunks con filtros?
- ¿Puedes hacer preguntas entre proyectos sin filtros?
- ¿Los filtros funcionan con consultas en lenguaje natural?

**Qué está pasando:**

- El pre-filtrado reduce el espacio de búsqueda antes de búsqueda de similitud
- Recuperación más dirigida → contexto más relevante → mejores respuestas
- Combina filtrado estructurado (metadatos) con búsqueda semántica (embeddings)
- Esto es **recuperación híbrida**

**Concepto clave:** El filtrado de metadatos es crucial para colecciones grandes de documentos. Previene que el LLM sea abrumado con contexto irrelevante.

---

### Actividad F — Evaluación y comparación RAG (opcional)

Ejecutar:

```bash
python examples/rag/06_rag_evaluation.py
```

**Lo que hace este script:**

1. Crea un **conjunto de prueba** de preguntas con respuestas conocidas (ground truth)
2. **Compara tres enfoques**:
   - **LLM Directo** (sin RAG, sin acceso a documentos)
   - **RAG con recuperación** (pipeline completo)
   - **Oráculo** (LLM con documentos enteros, no solo chunks)
3. **Evalúa cada enfoque**:
   - Precisión factual (¿la respuesta coincide con ground truth?)
   - Calidad de citaciones (¿se proveen fuentes y son correctas?)
   - Completitud (¿está incluida toda la info relevante?)
   - Tasa de alucinación (¿el LLM inventa información?)
4. Genera reporte de comparación: `outputs/04_rag_evaluation_report.html`

**Preguntas de prueba (ejemplos):**

```text
1. ¿Cuál fue la tasa de respuesta en el Proyecto A?
   Ground truth: 68%, superando meta del 65%
   
2. ¿Qué barreras de idioma se encontraron?
   Ground truth: Falta de materiales en lenguas indígenas en Proyecto B

3. ¿Qué proyecto tuvo conflictos de horario con temporadas agrícolas?
   Ground truth: Proyecto C

4. ¿Qué recomendaron los tres proyectos para participación comunitaria?
   Ground truth: Asociación temprana, materiales culturalmente apropiados, programación flexible
```

**Métricas de evaluación:**

| Métrica | Definición | Buen puntaje |
|--------|------------|------------|
| **Precisión** | % de respuestas factualmente correctas | >90% |
| **Tasa de citación** | % de respuestas con citaciones válidas | >80% |
| **Precisión de recuperación** | % de chunks recuperados que son relevantes | >70% |
| **Tasa de alucinación** | % de respuestas con hechos inventados | <5% |

**Resultados de ejemplo:**

```text
Enfoque          | Precisión | Citaciones | Alucinaciones
-----------------|----------|-----------|---------------
LLM Directo      | 25%      | 0%        | 60% (inventa detalles)
RAG (nuestro)    | 92%      | 88%       | 3%
Oráculo (todos)  | 95%      | 90%       | 2%
```

**Qué observar:**

- ¿Cuánto mejora RAG sobre el LLM directo?
- ¿Dónde falla RAG? (¿recuperación incorrecta? ¿generación incorrecta?)
- ¿Está RAG cerca del rendimiento del oráculo?
- ¿Qué tipos de preguntas son más difíciles para RAG?

**Concepto clave:** La evaluación revela dónde tu sistema RAG necesita mejora—problemas de recuperación vs. generación requieren diferentes soluciones.

---

## Problemas comunes y soluciones

**"La conversión de PDF tiene texto distorsionado"**
→ El PDF podría ser escaneado (basado en imagen). Usa herramienta OCR primero (Adobe Acrobat, Tesseract).

**"Los chunks son muy pequeños/muy grandes"**
→ Ajusta parámetros de chunking en `02_create_document_chunks.py`. Prueba rango 300-800 tokens.

**"La recuperación retorna chunks irrelevantes"**
→ Revisa puntajes de similitud. Si <0.6, aumenta k (recupera más chunks) o reformula consulta. Considera filtrado de metadatos.

**"El LLM alucina a pesar de RAG"**
→ Fortalece el prompt: "Responde SOLO del contexto proporcionado. Si no se encuentra, di 'No disponible en documentos.'" Considera usar Claude para mejor seguimiento de instrucciones.

**"El sistema es muy lento"**
→ Reduce k (menos chunks recuperados), cachea embeddings, o usa procesamiento por lotes para múltiples consultas.

**"Las respuestas carecen de detalle"**
→ Aumenta superposición de chunks (más contexto) o recupera más chunks (valor k más alto, ej., k=8).

**"Las comparaciones entre proyectos son débiles"**
→ Asegura que la recuperación abarque múltiples proyectos (aumenta k o usa metadatos para asegurar diversidad). Agrega instrucción: "Compara entre todos los proyectos disponibles."

**"El archivo de base de datos es muy grande"**
→ Usa una base de datos vectorial apropiada (Pinecone, Weaviate, Chroma) en lugar de CSV/pickle. Estas están optimizadas para embeddings.

---

## Modelo mental: Cuándo usar RAG

```text
¿Necesitas consultar información específica de proyectos?
├─ SÍ → Usa RAG
│   ├─ Enfoque en un solo proyecto? → Usa filtrado de metadatos
│   └─ Análisis entre proyectos? → Usa recuperación completa
└─ NO → Usa prompting directo del LLM

Tamaño de colección de documentos:
├─ 1-3 documentos → RAG es excesivo, usa prompting directo con texto completo
├─ 3-10 documentos → RAG es útil
└─ 10+ documentos → RAG es esencial

Tipos de consulta:
├─ Búsqueda factual (tasas de respuesta, fechas, números) → RAG sobresale
├─ Síntesis entre documentos → RAG + prompting cuidadoso
├─ Tareas creativas (lluvia de ideas, ideación) → LLM Directo
└─ Opinión/juicio → Híbrido (RAG para hechos, LLM para razonamiento)
```

---

## Lo que aprendiste

- ✅ Convertir PDFs a markdown para procesamiento LLM
- ✅ Crear chunks de documentos con metadatos
- ✅ Generar embeddings y construir bases de datos vectoriales
- ✅ Implementar sistemas de recuperación (búsqueda semántica)
- ✅ Construir pipelines RAG end-to-end
- ✅ Consultar conocimiento de proyectos con citaciones
- ✅ Usar filtrado de metadatos para recuperación dirigida
- ✅ Evaluar rendimiento de RAG vs. prompting directo
- ✅ Entender trade-offs: RAG vs. fine-tuning vs. prompting directo

---

## Recursos

- **Fundamentos RAG**: <https://www.pinecone.io/learn/retrieval-augmented-generation/>
- **Tutorial RAG LangChain**: <https://python.langchain.com/docs/use_cases/question_answering/>
- **Comparación Bases Datos Vectoriales**: <https://www.pinecone.io/learn/vector-database/>
- **Guía Embeddings OpenAI**: <https://platform.openai.com/docs/guides/embeddings>
- **Estrategias de Chunking**: <https://www.pinecone.io/learn/chunking-strategies/>
- **Evaluación RAG**: <https://docs.llamaindex.ai/en/stable/examples/evaluation/>
- **Procesamiento PDF**: <https://pymupdf.readthedocs.io/>

---

## Herramientas para explorar

**Bases de datos vectoriales:**

- **Pinecone**: Gestionado, escalable, configuración fácil
- **Weaviate**: Código abierto, rico en características
- **Chroma**: Ligero, bueno para prototipos
- **FAISS**: Librería de Facebook, rápida pero más bajo nivel

**Frameworks RAG:**

- **LangChain**: Completo, muchas integraciones
- **LlamaIndex**: Enfocado en indexación de documentos
- **Haystack**: Listo para producción, enfoque NLP

**Herramientas de evaluación:**

- **RAGAS**: Métricas de evaluación RAG
- **TruLens**: Observabilidad y evaluación RAG
- **LangSmith**: Plataforma de evaluación de LangChain

---

**¡Felicitaciones!** Has construido un sistema RAG completo para gestión de conocimiento de proyectos. Ahora puedes convertir cualquier colección de documentos en una base de conocimiento consultable que los LLMs pueden usar para proporcionar respuestas precisas y citadas a tus preguntas de investigación.

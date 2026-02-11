import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. CONFIGURAÇÃO INICIAL ---
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not GEMINI_KEY:
    print("⚠️ AVISO: GEMINI_API_KEY não encontrada!")
if not SUPABASE_URL:
    print("⚠️ AVISO: SUPABASE_URL não encontrada!")

genai.configure(api_key=GEMINI_KEY)
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ Erro ao conectar Supabase: {e}")
    supabase = None

# --- 2. DEFINIÇÃO DA APP FASTAPI ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuizRequest(BaseModel):
    query: str 

# --- 3. FUNÇÕES AUXILIARES ---
def buscar_contexto(pergunta_usuario):
    print(f"🔎 Buscando contexto para: '{pergunta_usuario}'...")
    if not supabase:
        return []
        
    try:
        embedding = genai.embed_content(
            model="models/gemini-embedding-001",
            content=pergunta_usuario,
            task_type="retrieval_query",
            output_dimensionality=768
        )
        vetor_pergunta = embedding['embedding']
        
        response = supabase.rpc(
            "match_documents",
            {
                "query_embedding": vetor_pergunta,
                "match_threshold": 0.5, 
                "match_count": 3
            }
        ).execute()

        return response.data
    except Exception as e:
        print(f"❌ Erro na busca vetorial: {e}")
        return []

# --- 4. ROTAS DA API ---

@app.get("/")
def home():
    return {"message": "API do PandAI está online e rodando! 🐼🚀"}

@app.post("/search-lessons")
def search_lessons_route(request: QuizRequest):
    contexto = buscar_contexto(request.query)
    return {"results": contexto}

@app.post("/generate-quiz-preview")
async def generate_quiz_route(request: QuizRequest):
    topic = request.query
    print(f"🚀 [API] Gerando Quiz sobre: {topic}")

    contexto = buscar_contexto(topic)

    if not contexto:
        return {
            "success": False, 
            "message": "Não encontramos conteúdo suficiente nas aulas para este tema."
        }

    texto_base = "\n\n".join([f"--- TRECHO DE AULA ---\n{item['content']}" for item in contexto])

    # CONFIGURAÇÃO JSON PARA O MODELO
    generation_config = {
        "temperature": 0.2, # Baixa temperatura para ser mais preciso
        "response_mime_type": "application/json", # Força resposta JSON nativa do Gemini 1.5
    }

    model = genai.GenerativeModel("models/gemini-2.5-flash", generation_config=generation_config)

    prompt = f"""
    Você é um sistema gerador de avaliações técnicas.
    Analise o contexto abaixo e gere um quiz técnico no formato JSON estrito.

    CONTEXTO:
    {texto_base}

    ESTRUTURA DE RESPOSTA OBRIGATÓRIA (JSON):
    {{
      "quiz_title": "Título criativo relacionado ao tema",
      "description": "Uma breve descrição do que será avaliado",
      "questions": [
        {{
          "content": "Enunciado da pergunta aqui?",
          "options": [
            {{ "content": "Opção A", "is_correct": false }},
            {{ "content": "Opção B (correta)", "is_correct": true }},
            {{ "content": "Opção C", "is_correct": false }},
            {{ "content": "Opção D", "is_correct": false }},
            {{ "content": "Opção E", "is_correct": false }}
          ]
        }}
      ]
    }}

    REGRAS:
    1. Crie exatamente 3 perguntas.
    2. Cada pergunta deve ter 5 alternativas.
    3. Apenas uma alternativa correta ("is_correct": true) por pergunta.
    4. Baseie-se APENAS no contexto fornecido.
    5. NÃO inclua markdown (```json), apenas o objeto JSON puro.
    """

    try:
        response = model.generate_content(prompt)
        print("✅ Quiz JSON gerado!")
        return {
            "success": True,
            "quiz_content": response.text # Agora será um JSON válido
        }
    except Exception as e:
        print(f"❌ Erro Gemini: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar quiz.")

# --- 5. INICIALIZAÇÃO ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Servidor iniciando na porta {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)

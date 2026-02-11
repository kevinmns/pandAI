import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from supabase import create_client, Client
from dotenv import load_dotenv

# --- 1. CONFIGURAÇÃO INICIAL ---
# Carrega variáveis de ambiente (localmente usa .env, na nuvem usa as vars do sistema)
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Validação básica para evitar erros silenciosos
if not GEMINI_KEY:
    print("⚠️ AVISO: GEMINI_API_KEY não encontrada!")
if not SUPABASE_URL:
    print("⚠️ AVISO: SUPABASE_URL não encontrada!")

# Configura clientes
genai.configure(api_key=GEMINI_KEY)
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"❌ Erro ao conectar Supabase: {e}")
    supabase = None

# --- 2. DEFINIÇÃO DA APP FASTAPI ---
app = FastAPI()

# Configura CORS (para seu frontend conseguir acessar)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Em produção, substitua "*" pela URL do seu site
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo de dados recebido do Frontend
class QuizRequest(BaseModel):
    query: str 

# --- 3. FUNÇÕES AUXILIARES ---
def buscar_contexto(pergunta_usuario):
    print(f"🔎 Buscando contexto para: '{pergunta_usuario}'...")
    if not supabase:
        return []
        
    try:
        # Gera embedding (vetor) da pergunta
        embedding = genai.embed_content(
            model="models/gemini-embedding-001",
            content=pergunta_usuario,
            task_type="retrieval_query",
            output_dimensionality=768
        )
        vetor_pergunta = embedding['embedding']
        
        # Busca no Supabase via RPC
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
    """
    Rota que gera o quiz com IA baseado no contexto das aulas.
    """
    topic = request.query
    print(f"🚀 [API] Gerando Quiz sobre: {topic}")

    # 1. Busca contexto
    contexto = buscar_contexto(topic)

    if not contexto:
        return {
            "success": False, 
            "message": "Não encontramos conteúdo suficiente nas aulas para este tema."
        }

    # 2. Prepara prompt
    texto_base = "\n\n".join([f"--- TRECHO DE AULA ---\n{item['content']}" for item in contexto])

    # 3. Chama o Gemini
    model = genai.GenerativeModel("models/gemini-2.5-flash") # Ajuste o modelo se necessário

    prompt = f"""
    ATUE COMO UM PROFESSOR DE TECNOLOGIA.
    Crie um Quiz Técnico curto baseado EXCLUSIVAMENTE no contexto abaixo.
    
    CONTEXTO:
    {texto_base}
    
    REGRAS:
    1. Crie 3 perguntas de múltipla escolha.
    2. Indique a resposta correta.
    3. Use formatação Markdown clara.
    """

    try:
        response = model.generate_content(prompt)
        return {
            "success": True,
            "quiz_content": response.text
        }
    except Exception as e:
        print(f"❌ Erro Gemini: {e}")
        raise HTTPException(status_code=500, detail="Erro interno ao gerar quiz.")

# --- 5. INICIALIZAÇÃO DO SERVIDOR ---
if __name__ == "__main__":
    # Pega a porta do ambiente (padrão Discloud/Heroku/Render) ou usa 8080
    port = int(os.environ.get("PORT", 8080))
    
    print(f"🚀 Servidor iniciando na porta {port}...")
    
    # IMPORTANTE: Passamos o objeto 'app' diretamente, não uma string.
    # Isso evita erros de "módulo não encontrado" se o nome do arquivo mudar.
    uvicorn.run(app, host="0.0.0.0", port=port)

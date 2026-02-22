"""
services/chat.py

RESPONSABILIDADE: Montar o agente de atendimento com RAG.

O QUE MUDOU EM RELAÇÃO À VERSÃO 0.3.x:
─────────────────────────────────────────────────────────────────
  REMOVIDO (depreciado/sumiu em 1.x):
    ✗ ConversationalRetrievalChain    → movido para langchain-classic
    ✗ ConversationBufferWindowMemory  → movido para langchain-classic
    ✗ create_history_aware_retriever  → sumiu do langchain.chains
    ✗ create_retrieval_chain          → sumiu do langchain.chains
    ✗ create_stuff_documents_chain    → sumiu do langchain.chains

  ADICIONADO (padrão oficial 1.x):
    ✓ create_agent()    → nova API padrão para agentes em LangChain 1.x
    ✓ init_chat_model() → nova forma de inicializar qualquer LLM
    ✓ @tool              → decorator para transformar função em ferramenta
    ✓ Histórico como mensagens (igual antes, isso não mudou)

CONCEITOS NOVOS:
─────────────────────────────────────────────────────────────────

1. create_agent()
   É a nova API padrão do LangChain 1.x para criar agentes.
   Substitui create_react_agent do LangGraph e o LCEL manual.
   Internamente usa LangGraph como runtime.

   Parâmetros principais:
   - model      → "provider:model" (ex: "google_genai:gemini-2.5-flash")
   - tools      → lista de ferramentas que o agente pode chamar
   - system_prompt → instruções de comportamento

2. init_chat_model()
   Nova forma padrão de inicializar um LLM em 1.x.
   Provider-agnostic: muda o provider trocando só a string.
   Ex: "google_genai:gemini-2.5-flash" → "anthropic:claude-3-5-sonnet"

3. @tool com response_format="content_and_artifact"
   O decorator @tool transforma uma função Python em uma ferramenta
   que o agente pode decidir chamar.
   response_format="content_and_artifact" → retorna texto E objetos
   (útil para RAG: texto para o LLM, documentos para rastreabilidade)

4. COMO O AGENTE "LEMBRA" A CONVERSA?
   O create_agent usa internamente o LangGraph com persistência.
   Passamos o histórico como lista de mensagens no invoke(),
   igual à abordagem anterior — isso não mudou.

FLUXO DO AGENTE:
─────────────────────────────────────────────────────────────────
  Usuário pergunta
        ↓
  Agente decide: "preciso buscar informações sobre isso?"
        ↓
  [Se sim] Chama a tool buscar_produtos(query)
        ↓
  Tool executa similarity_search no FAISS
        ↓
  Agente recebe os chunks e gera a resposta final
        ↓
  Resposta para o usuário
"""

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from config import GOOGLE_API_KEY, MODEL_NAME


def create_chat_agent(retriever):
    """
    Cria o agente de atendimento com acesso ao RAG via tool.

    POR QUE A TOOL É CRIADA AQUI DENTRO? (closure)
    ─────────────────────────────────────────────────
    O problema com variável global (_retriever = None) é que o @tool
    é executado no momento da importação do módulo — quando o retriever
    ainda não existe. O type checker reclama, e em runtime pode quebrar.

    A solução é criar a tool DENTRO da função, usando closure:
    a função interna "buscar_produtos" captura o "retriever" do escopo
    externo automaticamente — ele já existe e tem tipo garantido.

    CLOSURE = função que "lembra" variáveis do escopo onde foi criada.

    DIFERENÇA CHAVE em relação à chain antiga:
    - Antes: fluxo fixo (sempre busca → sempre responde)
    - Agora: agente DECIDE quando buscar (mais inteligente)
    """

    # ✅ Tool criada como closure: captura `retriever` do escopo acima.
    # Agora o type checker sabe que retriever nunca é None aqui.
    @tool(response_format="content_and_artifact")
    def buscar_produtos(query: str):
        """
        Busca produtos e informações da loja SportShop relevantes para a query.
        Use essa ferramenta para responder perguntas sobre produtos,
        preços, tamanhos, cores, frete, pagamento e trocas.
        """
        docs = retriever.invoke(query)  # retriever garantidamente não é None

        conteudo = "\n\n".join(
            f"[{doc.metadata.get('source', 'loja')}]\n{doc.page_content}"
            for doc in docs
        )

        return conteudo, docs  # (texto para o LLM, objetos para rastreabilidade)

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    # init_chat_model: nova forma padrão em 1.x — formato "provider:model"
    model = init_chat_model(MODEL_NAME)

    agent = create_agent(
        model=model,
        tools=[buscar_produtos],
        system_prompt=(
            "Você é um assistente de atendimento da SportShop, loja de artigos esportivos. "
            "Seja simpático, objetivo e ajude o cliente a encontrar o produto certo.\n\n"
            "Regras:\n"
            "- Use a ferramenta buscar_produtos para responder sobre produtos, preços, "
            "frete, pagamento e trocas\n"
            "- Sempre mencione preço e disponibilidade ao recomendar produtos\n"
            "- Se não encontrar a informação, oriente o cliente a contatar "
            "o WhatsApp: (11) 99999-9999"
        ),
    )

    return agent


def chat(agent, pergunta: str, historico: list) -> str:
    """Envia uma pergunta para o agente e retorna a resposta como texto limpo."""
    mensagens = historico + [HumanMessage(content=pergunta)]
    result = agent.invoke({"messages": mensagens})
    ultima_mensagem = result["messages"][-1]

    # .text é o atributo padrão do LangChain 1.x para extrair texto puro
    # de qualquer mensagem, independente de ser string ou lista de content blocks
    return ultima_mensagem.text

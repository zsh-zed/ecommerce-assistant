import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from config import GOOGLE_API_KEY, MODEL_NAME


def create_chat_agent(retriever):

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

        return conteudo, docs

    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

    model = init_chat_model(MODEL_NAME)

    agent = create_agent(
        model=model,
        tools=[buscar_produtos],
        system_prompt=(
            "Você é um assistente de atendimento da SportShop, loja de artigos esportivos. "
            "Seja simpático, objetivo e ajude o cliente a encontrar o produto certo.\n\n"
            "Regras:\n"
            "- REGRA CRÍTICA: Responda APENAS com informações do contexto abaixo. Nunca invente preços, prazos ou condições."
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

    return ultima_mensagem.text

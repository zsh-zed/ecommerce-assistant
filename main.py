from langchain_core.messages import AIMessage, HumanMessage

from services.chat import chat, create_chat_agent
from services.indexer import get_retriever


def main():
    print("=" * 50)
    print("  🛍️  SportShop — Assistente de Atendimento")
    print("       LangChain 1.x + Gemini 2.5 Flash")
    print("=" * 50)
    print()

    # Passo 1: Prepara o RAG (igual antes)
    retriever = get_retriever()

    # Passo 2: Cria o agente (novo em 1.x)
    agent = create_chat_agent(retriever)

    # Passo 3: Histórico manual de mensagens (igual antes)
    historico = []
    MAX_HISTORICO = 10

    print("💬 Assistente pronto! Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("Você: ").strip()

        if not pergunta:
            continue

        if pergunta.lower() in ["sair", "exit"]:
            print("Assistente: Até logo! 👋")
            break

        resposta = chat(agent, pergunta, historico)
        print(f"\nAssistente: {resposta}\n")

        historico.append(HumanMessage(content=pergunta))
        historico.append(AIMessage(content=resposta))

        # Janela deslizante
        if len(historico) > MAX_HISTORICO:
            historico = historico[-MAX_HISTORICO:]


if __name__ == "__main__":
    main()

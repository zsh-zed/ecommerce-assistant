# 🛍️ SportShop — Assistente de Atendimento

Assistente de atendimento para e-commerce usando **LangChain 1.x**, **RAG com FAISS** e **Gemini 2.5 Flash**.

---

## Stack

- **LangChain 1.x** — framework de IA
- **FAISS** — busca semântica local
- **Gemini 2.5 Flash** — LLM via Google AI Studio
- **Gemini Embedding 001** — embeddings

## Como rodar

```bash
# 1. Ambiente virtual
python -m venv venv && source venv/bin/activate

# 2. Dependências
pip install -r requirements.txt

# 3. Criar .env
echo "GOOGLE_API_KEY=sua_chave_aqui" > .env

# 4. Rodar
python main.py
```

> API key em: [aistudio.google.com](https://aistudio.google.com) → Get API Key

## Estrutura

```
├── data/
│   ├── catalogo.csv      # Produtos
│   └── loja_info.txt     # Frete, pagamento, trocas
├── services/
│   ├── indexer.py        # RAG: carrega dados e indexa no FAISS
│   └── chat.py           # Chain: RAG + histórico + LLM
├── config.py             # Modelo e configurações
└── main.py               # Loop de conversa
```

## O que você aprendeu

| Conceito            | Onde         |
| ------------------- | ------------ |
| Document Loaders    | `indexer.py` |
| Text Splitters      | `indexer.py` |
| Embeddings + FAISS  | `indexer.py` |
| RAG                 | `chat.py`    |
| Memória de conversa | `main.py`    |
| LCEL                | `chat.py`    |

## Próximo passo

**LangGraph** — transformar esse assistente em um agente com fluxos condicionais, múltiplas ferramentas e estado persistente.

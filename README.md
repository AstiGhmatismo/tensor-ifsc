# Tensor-IFSC

Assistente acadêmico experimental para estudantes do IFSC Câmpus Garopaba. A aplicação combina uma interface web responsiva, API FastAPI, recuperação de trechos documentais (RAG), embeddings locais e uma LLM em endpoint compatível com OpenAI.

**Atenção:** esta versão é uma base funcional de desenvolvimento. Nenhum documento oficial do câmpus está incluído. Não use para decisões sobre prazos, bolsas ou direitos acadêmicos antes de indexar e revisar fontes oficiais.

## Executar com Docker

1. Copie `.env.example` para `.env`.
2. Altere `ADMIN_TOKEN` para um segredo aleatório.
3. Configure o endpoint e o identificador do modelo em `LLM_BASE_URL` e `LLM_MODEL`.
4. Coloque documentos autorizados em `data/documents/`.
5. Execute `docker compose up --build`.
6. Acesse `http://localhost:8000`.

O servidor LLM deve estar acessível a partir do container. Para LM Studio no host, o exemplo usa `host.docker.internal:1234`.

## Indexação

```bash
curl -X POST http://localhost:8000/api/admin/ingest -H "X-Admin-Token: SEU_TOKEN"
```

São aceitos PDF, DOCX, TXT e Markdown. A coleção vetorial é persistida em volume Docker.

## API

- `GET /api/health`
- `POST /api/chat` com JSON `{"question":"..." }`
- `POST /api/admin/ingest` protegido por token
- `GET /docs` documentação OpenAPI

## Segurança e limitações

Use apenas documentos públicos ou autorizados; não indexe dados pessoais. Antes de disponibilizar em rede, adicione HTTPS, autenticação dos estudantes, rate limiting, monitoramento e revisão de segurança. RAG reduz, mas não elimina alucinações. O campo de fontes representa trechos recuperados, não uma garantia de correção. WhatsApp não está implementado.

## Desenvolvimento

```bash
python -m venv .venv
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest
```

Veja [docs/VALIDATION.md](docs/VALIDATION.md) para o plano de testes.

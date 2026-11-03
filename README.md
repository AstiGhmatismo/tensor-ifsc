# Tensor-IFSC

Assistente acadêmico experimental para estudantes do IFSC Câmpus Garopaba. Interface web responsiva, API FastAPI e recuperação de informações em documentos institucionais versionados no repositório (RAG leve por relevância lexical).

## Modelo de linguagem

O projeto utiliza **Gemini 3.5 Flash-Lite** pela API compatível com OpenAI do Google. O modelo foi escolhido por ser adequado a perguntas curtas e por oferecer faixa gratuita na API, sujeita a limites e disponibilidade do Google. É necessária uma chave criada no Google AI Studio; ela deve ser cadastrada como segredo `LLM_API_KEY` no Render e nunca no GitHub.

## Executar localmente

1. Copie `.env.example` para `.env`.
2. Configure `LLM_API_KEY`.
3. Instale as dependências: `pip install -r requirements.txt`.
4. Execute: `uvicorn app.main:app --reload`.

## Deploy no Render

O serviço usa Docker e escuta a porta definida por `PORT`. Configure no Render:
- `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`
- `LLM_MODEL=gemini-3.5-flash-lite`
- `LLM_API_KEY` como segredo.

O corpus fica versionado em `data/knowledge/`; não depende do filesystem efêmero nem de disco persistente. O plano gratuito pode suspender o serviço após inatividade e causar demora na primeira abertura.

## Fontes e limites

O conteúdo inicial foi compilado de páginas oficiais do IFSC Câmpus Garopaba e mantém o link da fonte em cada arquivo. Prazos, editais e calendários mudam; confirme sempre no portal oficial. A base inicial é pequena e não substitui os setores institucionais. Não envie dados pessoais ao chatbot.

## API

- `GET /api/health`
- `POST /api/chat` com JSON `{"question":"..." }`
- `GET /docs` documentação OpenAPI.

## Testes

Execute `pytest`. Não declare desempenho, precisão ou validação institucional sem testes correspondentes.

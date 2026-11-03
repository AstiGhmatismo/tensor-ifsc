# Validação e próximos passos

Os testes desta primeira entrega ainda precisam ser executados em ambiente com dependências instaladas. A existência de código não equivale a validação funcional.

## Cenários obrigatórios

1. Pergunta cuja resposta consta em documento oficial indexado: verificar precisão e fonte.
2. Pergunta sem evidência na base: o modelo deve declarar que não encontrou informação.
3. Edital vencido: verificar se o sistema evita afirmar que está vigente.
4. API LLM desligada: deve retornar erro controlado.
5. Ingestão sem token ou com token incorreto: deve retornar 401.
6. Teste responsivo em larguras de 360, 768 e 1280 px.

## Métricas

- Recall@5 da recuperação em conjunto de perguntas rotuladas.
- Precisão das fontes citadas.
- Taxa de respostas sem suporte documental.
- Latência mediana e p95.
- Disponibilidade da API.

Não publique métricas sem executar os experimentos. A próxima etapa recomendada é acrescentar autenticação estudantil, metadados de vigência e URL oficial, reranking, testes automatizados de integração e revisão de segurança.

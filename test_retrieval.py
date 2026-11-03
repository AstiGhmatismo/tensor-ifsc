from app.main import retrieve, tokenize


def test_tokenize_portuguese():
    assert "garopaba" in tokenize("IFSC Câmpus Garopaba")


def test_retrieve_finds_relevant_document():
    docs = [
        {"title": "Assistência", "source": "https://ifsc.edu.br", "text": "O PAEVS oferece auxílio permanência e auxílio emergencial."},
        {"title": "Biblioteca", "source": "https://ifsc.edu.br", "text": "O sistema de bibliotecas permite consultar o acervo."},
    ]
    result = retrieve("Como funciona o auxílio permanência?", docs)
    assert result
    assert result[0]["title"] == "Assistência"


def test_retrieve_empty_for_unrelated_query():
    docs = [{"title": "Calendário", "source": "", "text": "Datas acadêmicas do câmpus."}]
    assert retrieve("receita de bolo de chocolate", docs) == []

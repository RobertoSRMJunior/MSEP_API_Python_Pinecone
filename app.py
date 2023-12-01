import os

import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings import OpenAIEmbeddings


from flask import (Flask, redirect, render_template, request,
                   send_from_directory, url_for)

app = Flask(__name__)


# Carrega as chaves de API do ambiente
OPENAI_API_KEY = "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"
PINECONE_API_KEY = "74a1af62-50b3-4741-b340-4833aafda023"


# Inicia a conexão com o Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment='gcp-starter'
)

# Define o nome do índice Pinecone
index_pinecone = 'jarvis'


@app.route("/pergunta", methods=["POST"])
def search():
    # Extrai a pergunta da requisição JSON
    question = request.json["question"]

    # Cria um objeto OpenAIEmbeddings para gerar embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    # Cria um objeto Pinecone a partir do índice existente
    docsearch = Pinecone.from_existing_index(index_pinecone, embeddings)

    # Instancia o modelo ChatOpenAI e a cadeia de QA
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Busca os documentos semelhantes à pergunta
    docs = docsearch.similarity_search(question)

    # Executa a cadeia de QA com os documentos recuperados
    resposta = chain.run(input_documents=docs, question=question)

    # Retorna a resposta da cadeia de QA
    return {"resposta": resposta}




@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))


if __name__ == '__main__':
   app.run()

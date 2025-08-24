from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from rag import RAGSystem

app = Flask(__name__)
rag_system = RAGSystem()  # Initialize RAG


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    def generate():
        # Send "thinking" state immediately
        yield '{"status": "thinking"}\n'

        # Process query
        try:
            response = rag_system.query(question)
            yield jsonify({'status': 'complete', 'response': response}).get_data(as_text=True) + '\n'
        except Exception as e:
            yield jsonify({'status': 'error', 'error': str(e)}).get_data(as_text=True) + '\n'

    return Response(stream_with_context(generate()), mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

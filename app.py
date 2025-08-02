from flask import Flask, render_template, request, jsonify
from main import FinancialRAGSystem
from question_history import QuestionHistory
import os

app = Flask(__name__)
rag_system = FinancialRAGSystem()
question_history = QuestionHistory()

@app.route('/')
def home():
    # Add some example questions if history is empty
    if not question_history.get_financial_questions():
        example_questions = [
            ("What was Apple's revenue in the last quarter?", "Based on the financial report, Apple reported revenue of $81.8 billion for Q3 2023"),
            ("What is the profit margin trend?", "The profit margin has shown consistent growth, with gross margin at 44.5% in the most recent quarter"),
            ("Explain the cash flow position", "Apple maintains a strong cash position with operating cash flow of $28.7 billion"),
            ("What are the R&D expenses?", "Research and Development expenses were $22.61 billion, representing 7.8% of revenue"),
            ("How much is the dividend payout?", "Apple pays a quarterly dividend of $0.24 per share")
        ]
        for question, response in example_questions:
            question_history.add_question('financial', question, response)
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form.get('query')
    try:
        if not query:
            return jsonify({'response': 'Please provide a query'}), 400
        response = rag_system.analyze_query(query)
        question_history.add_question('financial', query, response)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'response': f'Error processing query: {str(e)}'}), 500

@app.route('/market_trends', methods=['POST'])
def market_trends():
    symbol = request.form.get('symbol')
    timeframe = request.form.get('timeframe', '1d')
    try:
        if not symbol:
            return jsonify({'response': 'Please provide a stock symbol'}), 400
        analysis, graph_data = rag_system.analyze_market_trends(symbol, timeframe)
        question_history.add_question('market', f"Market trends for {symbol} ({timeframe})", analysis)
        return jsonify({
            'response': analysis,
            'graphData': {
                'dates': graph_data['dates'].tolist(),
                'prices': graph_data['prices'].tolist(),
                'volume': graph_data['volume'].tolist()
            }
        })
    except Exception as e:
        return jsonify({'response': f'Error analyzing market trends: {str(e)}'}), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'response': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'response': 'Internal server error occurred'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    history = question_history.get_history()
    return jsonify({'history': history})

@app.route('/financial_questions', methods=['GET'])
def get_financial_questions():
    questions = question_history.get_financial_questions()
    return jsonify({'financial_questions': questions})

if __name__ == '__main__':
    # Load financial data on startup
    data_dir = "financial_data"
    try:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        rag_system.load_financial_data(data_dir)
    except Exception as e:
        print(f"Warning: Could not load financial data: {str(e)}")
    app.run(debug=True)

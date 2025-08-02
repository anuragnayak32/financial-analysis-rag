from flask import Flask, render_template, request, jsonify
from main import FinancialRAGSystem
import os

app = Flask(__name__)
rag_system = FinancialRAGSystem()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    query = request.form.get('query')
    try:
        if not query:
            return jsonify({'response': 'Please provide a query'}), 400
        response = rag_system.analyze_query(query)
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

from collections import deque
from datetime import datetime

class QuestionHistory:
    def __init__(self, max_size=100):
        self.history = deque(maxlen=max_size)

    def add_question(self, question_type: str, query: str, response: str):
        self.history.appendleft({
            'timestamp': datetime.now().isoformat(),
            'type': question_type,  # 'financial' or 'market'
            'query': query,
            'response': response
        })

    def get_history(self):
        return list(self.history)
        
    def get_financial_questions(self):
        return [item for item in self.history if item['type'] == 'financial']

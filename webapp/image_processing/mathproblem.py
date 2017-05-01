class MathProblem:

    def __init__(self, question, response):
        self.question = question
        self.response = response

    def get_question(self):
        """Question in string format  ex. '9x3' """
        return self.question

    def get_response(self):
        """Submitted answer in string format"""
        return self.response

    def get_answer(self):
        """ Returns int of calculated answer to question
            or None if there is no answer
        """
        operators = ["x","-","+"]
        for operand in operators:
            if self.question.find(operand)  == -1:
                continue
            else:
                temp = self.question.split(operand)
                if temp[0].isdigit() and temp[1].isdigit():                    
                    if operand == "x":
                        return int(temp[0]) * int(temp[1])
                    elif operand == "-":
                        return int(temp[0]) - int(temp[1])
                    elif operand == "+":
                        return int(temp[0]) + int(temp[1])
                else:
                    return None
        return None

### FUNCTIONS FOR INTRODUCTION

### user preference class ###
class UserPreference:

    def __init__(self):
        self.food = []
        self.price_range = []
        self.area = []

    def addFoodPreference(self, values):

        self.food.extend(values)

        return None

    def addPricePreference(self, values):
        self.price_range.extend(values)

        return None

    def addAreaPreference(self, values):
        self.area.extend(values)

        return None

    def getPreferences(self):
        preferences = {"food": self.food, "price_range": self.price_range, "area": self.area}

        return preferences

class conversation:
    sentences = {
        "hello1": "Welcome "

    }

    def __init__(self):
        self.phase = 'Hello'
        self.act = 'ack'

    def getNextSentence(self, sentence, phase, userPreferences, query):
        if(phase == 'Hello'):
            
            return(sentences("hello1"))

    def getDialogAct(self, sentence):





while():








### FUNCTIONS FOR INTRODUCTION

### user preference class ###
class UserPreference:

    def __init__(self):
        self.food = ['italian']
        self.price_range = ['cheap']
        self.area = ['south']
        self.preferences = {"food": self.food, "price_range": self.price_range, "area": self.area}
        self.suggestion = None

    def updatePreferences(self, keys, values):

        return None

    def getPreferences(self):

        return preferences

    def clearPreferences(self):

        return None

# conversation class
class conversation:
    sentences = {
        "hello1": "Hello! Welcome to the Ambrosia restaurant system. What kind of restaurant are you looking for?",
        "hello2": "Hi there! You can search for restaurants by area, price range or food type. What would you like?",
        "empty1": "Sorry, no restaurants matching your criteria were found. Would you like to try something else?",
        "repeat1": "Oh sorry. I'm gonna repeat that.",
        "noise1": "I'm sorry, I didn't get that. Could you repeat?",
        "bye1": "Goodbye!",
        "bye2": "Bye, maybe next time",
        "restart1": "Ok! Let's try again!",
        "thankyou1": "You're welcome!",
        "ack1": "So, what can I help you with?",
        "request1": "Please, tell me what kind of restaurant you are looking for.",
        "inform1": "Here's the restaurant's information: .",
        "area1": "In which area would you like to go?",
        "pricerange1": "What kind of price range do you prefer?",
        "type1": "What type of food would you like?",
        "suggest1": "What about ",
        "confirm1": "Yes,that is correct."
        "deny1": "No."

    }

    def __init__(self):
        self.phase = 'gathering'
        self.act = 'start'
        self.Message = 'Nothing'
        self.userPreferences = UserPreference
        self.query = []
        self.madeSuggestion = False
        self.infoGiven = False
        self.sentences = {
        "hello1": "Welcome ",
        "repeat1": 'okokok'
    }

    def getdialogact(self, sentence):
        dialogact = 'inform'
        return dialogact

    def getMatches(self, sentence, key='area'):

        return({'area': 'south'})

    def query(self, data, preferences):
        queriedlist = ['macdonalds', 'kfc']
        return(queriedlist)

    def getRandom(self, restaurantSubset):
        restaurant = {'name': 'macdonalds',
                      'price': 'cheap',
                      'area': 'south',
                      'food': 'fast',
                      'phone': '04905903540',
                      'addr': 'my house',
                      'postcode': '2265dd'}
        return(restaurant)



    def getNextSentence(self, restaurantSubset, sentence=None):

        # the first message
        if((self.phase == 'Hello') and (sentence is None)):
            self.Message = self.sentences["hello1"]
            return(self.Message)

        # get dialog act
        self.act = self.getdialogact(sentence)

        # repeat message
        if(self.act == 'repeat'):
            return(self.sentences["repeat1"] + self.Message)

        # noise message
        if(self.act == 'null'):
            self.Message = self.sentences['noise1']
            return(self.Message)

        # bye message
        if(self.act == 'bye'):
            return(self.sentences['bye1'])

        # restart message
        if(self.act == 'restart'):
            self.act = 'start'
            self.phase = 'Hello'
            self.Message = None
            return(self.sentences['restart1'])

        # thankyou message
        if(self.act == 'thankyou'):
            return(self.sentences['thankyou1'])

        # hello message
        if(self.phase =='Hello'):
            if(self.act == 'hello'):
                return(self.sentences["hello2"])
            if(self.act == 'ack' or self.act == 'affirm'):
                return(self.sentences['ack1'])
            if(self.act == 'confirm'):
                return(self.sentences['request1'])
            if(self.act == 'deny'):
                return(self.sentences['bye2'])
            if(self.act == 'reqmore' or self.act == 'request'):
                return(self.sentences['noise1'])
            if(self.act == 'inform' or self.act == 'deny' or self.act == 'reqalts'):
                self.phase = 'gathering'


        if(self.phase =='gathering'):
            self.userPreferences.updatePreferences(getMatches(sentence))
            # update subset
            if(restaurantSubset < 1):
                self.userPreferences.clearPreferences()
                return(self.sentences['empty1'])

            if(restaurantSubset < 5):
                self.phase = 'suggestions'

            if(restaurantSubset > 5):
                if(self.act == 'hello' or self.act =='reqmore'):
                    return(self.sentences['noise1'])

                if(self.act == 'inform'):
                    if (self.userPreferences['area'] == []):
                        return (self.sentences['area1'])
                    if (self.userPreferences['price_range'] == []):
                        return (self.sentences['pricerange1'])
                    if (self.userPreferences['type'] == []):
                        return (self.sentences['type1'])
                    else:
                        self.phase = 'suggestions'
                if(self.act == 'ack' or self.act == 'affirm' or self.act == 'negate' or self.act == 'reqmore'):
                    return(self.Message)

                if(self.act =='confirm'):
                    self.phase = 'confirm'

                if(self.act =='deny'):
                    self.phase = 'confirm'

                if(self.act == 'reqalts'):
                    if (self.userPreferences['area'] == []):
                        return (self.sentences['area1'])
                    if (self.userPreferences['price_range'] == []):
                        return (self.sentences['pricerange1'])
                    if (self.userPreferences['type'] == []):
                        return (self.sentences['type1'])
                    else:
                        self.phase = 'suggestions'


        if(self.phase == 'suggestions'):
            if(self.madeSuggestion == False
                    or self.act == 'deny'
                    or self.act == 'negate'
                    or self.act == 'reqalts'
                    or self.act == 'reqmore'):

                self.suggestion = self.getRandom(restaurantSubset)

                return(self.sentences['suggest1']
                       + self.suggestion['name']
                       + " it's"
                       + self.suggestion['price']
                       + " and its in the "
                       + self.suggestion['area'] )

            if(self.act == 'ack' or self.act == 'affirm'):
                if(self.infoGiven == False):
                    return(self.sentences['inform1']
                           + self.suggestion['name']
                           + self.suggestion['addr']
                           + self.suggestion['postcode']
                           + self.suggestion['phone'])
                if(self.infoGiven == True):
                    self.phase = 'goodbye'

            if(self.act == 'confirm'):
                # get match and check
                if(True):
                    return(self.sentences['confirm1'])
                if(False):
                    return(self.sentences['deny1'])

            if(self.act == 'hello'):
                return(self.sentence['noise1'])

            if(self.act == 'inform'):
                self.phase = 'confirm'

            if(self.act == 'request'):
                # get match and return info
                return("this doesnt work yet")


        if(self.phase == 'confirm'):
            return('ok')


        if(self.phase == 'goodbye'):
            return('bye')






convo = conversation()
sent = 'Hello I want to have italian food in center pls'
print(convo.getNextSentence(sent))








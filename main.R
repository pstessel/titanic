Titanic.path <- "https://github.com/pstessel/Titanic.git/"
train.data.file <- "train.csv"
test.data.file <- "test.csv"
missing.types <- c("NA", "")
train.column.types <- c('integer', #PassengerId)
                        'factor', # Survived
                        'factor', # Pclass
                        'character', # Name
                        'factor', # Sex
                        'numeric', # Age
                        'integer', # SibSp
                        'integer', # Parch
                        'character', # Ticket
                        'numeric', # Fare
                        'character', # Cabin
                        'factor', # Embarked
)
test.column.types <- train.column.types[-2]
                          
                          
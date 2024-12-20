### Kaggle Titanic

Classic kaggle introductory competition to predict who survived the titanic.

Task: Simple binary class prediction.

Data:
- PassengerId: Index
- Survived: Target (not present in test set)
  - 0 is dead, 1 is alive

- Pclass: ticket class - ordinal
- Name: - string data, can create:
  - Title: - nominal
  - Family Size: number of matching surnames - discrete
- Sex: - binary
- Age: - discrete (20% NaN)
- SibSp: # of siblings/spouses - discrete
- Parch: # of parents/children - discrete
- Ticket: string data - nominal (681 categories in train set):
  - Definitely some patterns to get out
- Fare: - continuous
- Cabin: - string data (80% NaN):
  - Floor: - ordinal
- Embarked: port of Embarkation - nominal (couple NaN)
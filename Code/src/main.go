package main

import (
	"fmt"
)

func main() {

	// read the data and tokenize the
	path := "C:/Users/myrkul/Documents/TxtClass/Code/dataset/corpus_AGNews.csv"
	Documents := TokenizeDataset(path)
	vocab := InitVocab(Documents)
	ShuffelDataset(Documents)
	classes := []string{"Sports", "World", "Business", "Science"}
	model := InitModel(classes, vocab, .01)

	// Train
	epochs := 20
	trainingSize := len(Documents) / 2
	trainingSet := Documents[:trainingSize]
	fmt.Printf("Starting the stochastic training cycle of %v epochs\n", epochs)
	model.StocTrain(trainingSet, epochs)
	fmt.Println("Training Completed...")

	// Measure Accuracy
	testSet := Documents[trainingSize:]
	fmt.Println("Calculating model accuracy")
	model.MeasureAccuracy(testSet)
	fmt.Printf("Accuracy after %d epochs of Stochastic training:\t%.3f", epochs, model.Accuracy*float64(100))
}

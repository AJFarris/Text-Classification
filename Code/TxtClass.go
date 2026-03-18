package main

import (
	"fmt"
	"math"
	"os"
	//"slices"
	"strings"
	"unicode"
)

const SimThreshold float64 = .1

var Syms = []string{"<", ">", "*", "#"}

type Index map[string]int
type TermFreq map[string]float64
type InvDocFreq map[string]float64
type DocVector map[int]float64

type Cluster struct {
	SeedIndex int   //Document that starts the cluster
	Members   []int // Indexes of the docs that are part of the cluster
}

type LogisticRegression struct {
	Weights      [][]float64 //
	Bias         []float64
	NumClasses   int
	NumFeatures  int
	LearningRate int
}

func CleanDataset(file string) (dataset [][]string, voc Index, err error) {

	rawData, err := os.ReadFile(file)
	if err != nil {
		fmt.Println(err)
		return nil, nil, err
	}

	rawDocs := strings.Split(string(rawData), "\n")
	dataset = make([][]string, 0, len(rawDocs))
	voc = make(Index)

	for _, doc := range rawDocs {

		var document []string
		seen := make(map[string]bool)
		for _, word := range strings.Fields(doc) {

			word = strings.ReplaceAll(word, "'", "")
			word = strings.Map(func(r rune) rune {
				if unicode.IsLetter(r) || unicode.IsDigit(r) {
					return r
				}
				return -1
			}, word)

			clean := strings.ToLower(strings.TrimSpace(word))
			if clean == "" {
				continue
			}

			document = append(document, clean)
			if !seen[clean] {
				voc[clean]++
				seen[clean] = true
			}
		}
		dataset = append(dataset, document)
	}
	return dataset, voc, nil
}

func BuildIndex(df Index) map[string]int {

	index := make(map[string]int)
	i := 0
	for word := range df {
		index[word] = i
		i++
	}
	return index
}

func ComputeIDF(df Index, n int) InvDocFreq {

	idf := make(InvDocFreq)
	for word, count := range df {
		idf[word] = math.Log(float64(1+n) / float64(1+count))
	}
	return idf
}

func ComputeTF(doc []string) map[string]float64 {

	tf := make(map[string]float64)
	for _, word := range doc {
		tf[word]++
	}

	for word := range tf {
		tf[word] /= float64(len(doc))
	}
	return tf
}

func ComputeSparseVector(tf TermFreq, idf InvDocFreq, index Index) DocVector {

	doc := make(DocVector)

	for word, freq := range tf {
		ind := index[word]
		doc[ind] = freq * idf[word]
	}
	return doc
}

func NormalizeTFIDF(doc DocVector) DocVector {

	sum := float64(0)
	for _, value := range doc {
		sum += value * value
	}

	eucLen := math.Sqrt(sum)
	if eucLen == 0 {
		return doc
	}

	for word, value := range doc {
		doc[word] = value / eucLen
	}
	return doc
}

func SimilarityValue(docA, docB DocVector) float64 {

	dotProduct := 0.0
	if len(docA) < len(docB) {
		for termIndex, weightA := range docA {
			if weightB, exists := docB[termIndex]; exists {
				dotProduct += weightA * weightB
			}
		}
	} else {

		for termIndex, weightB := range docB {
			if weightA, exists := docA[termIndex]; exists {
				dotProduct += weightA * weightB
			}
		}
	}
	return dotProduct
}

func main() {

	// working so far. I need to confirm why the TFIDF key/values are not in the same spot when printed every time. Maybe we need to add a func to populate unfound words with a 0 val?
	data, df, err := CleanDataset("C:/Users/myrkul/Downloads/dataset_emails/emails.txt")
	if err != nil {
		fmt.Println(err)
		return
	}

	idf := ComputeIDF(df, len(data))
	index := BuildIndex(df)
	dataTFIDF := []DocVector{}
	for _, doc := range data {
		tf := ComputeTF(doc)
		dataTFIDF = append(dataTFIDF, ComputeSparseVector(tf, idf, index))
	}
	dataNormalized := []DocVector{}
	for _, doc := range dataTFIDF {
		norm := NormalizeTFIDF(doc)
		dataNormalized = append(dataNormalized, norm)
	}

}

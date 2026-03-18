package main

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

type Vocabulary struct {
	NumDocs  int
	NumWords int
	IDF      []float64
	Index    map[string]int
	Tokens   []string
	DF       map[string]int
}

func BuildDF(docs []Document) (df map[string]int) {

	df = make(map[string]int)
	for _, doc := range docs {
		for token := range doc.TFMap {
			df[token]++
		}
	}
	return df
}

func BuildIndex(df map[string]int) (tokens []string, index map[string]int) {

	tokens = make([]string, 0, len(df))
	for token := range df {
		tokens = append(tokens, token)
	}

	sort.Slice(tokens, func(i, j int) bool {
		if df[tokens[i]] == df[tokens[j]] {
			return tokens[i] < tokens[j]
		}
		return df[tokens[i]] > df[tokens[j]]
	})

	index = make(map[string]int)
	for i, token := range tokens {
		index[token] = i
	}
	return
}

func (v *Vocabulary) BuildIDF() []float64 {

	idf := make([]float64, len(v.Tokens))
	numerator := float64(v.NumDocs + 1)

	for i, token := range v.Tokens {
		denom := float64(v.DF[token] + 1)
		idf[i] = math.Log(numerator/(denom)) + 1
	}
	return idf
}

func InitVocab(docs []Document) Vocabulary {

	v := Vocabulary{
		NumDocs: len(docs),
		DF:      BuildDF(docs),
	}
	v.Tokens, v.Index = BuildIndex(v.DF)
	v.IDF = v.BuildIDF()
	v.NumWords = len(v.Tokens)
	return v
}

func (v *Vocabulary) VectorizeDoc(d Document) map[int]float64 {

	vect := make(map[int]float64)
	for token, tf := range d.TFMap {
		index, ok := v.Index[token]
		if !ok {
			continue
		}
		idf := v.IDF[index]
		vect[index] = tf * idf
	}
	return vect
}

func ShuffelDataset(docs []Document) {

	randomizer := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range docs {
		j := randomizer.Intn(i + 1)
		docs[i], docs[j] = docs[j], docs[i]
	}
}

package main

import (
	"math"
)

type Model struct {
	Accuracy  float64
	Classes   []string
	Vocab     Vocabulary
	Weights   [][]float64
	Biases    []float64
	LearnRate float64
}

type Prediction struct {
	TF_IDF    map[int]float64
	RawScores []float64
	Probs     []float64
	TrueClass int
	PredClass int
}

func InitModel(classes []string, v Vocabulary, lr float64) (m Model) {

	wordCount := v.NumWords
	classCount := len(classes)
	weights := make([][]float64, classCount)
	biases := make([]float64, classCount)
	for class := 0; class < classCount; class++ {
		weights[class] = make([]float64, wordCount)
	}

	m = Model{
		Classes:   classes,
		Vocab:     v,
		Weights:   weights,
		LearnRate: lr,
		Biases:    biases,
	}
	return
}

// Inference Pipeline
func (m *Model) CalcLogits(vec map[int]float64) (scores []float64) {

	scores = make([]float64, len(m.Biases))
	for class, classBias := range m.Biases {
		logit := 0.0
		for index, tfidf := range vec {
			logit += m.Weights[class][index] * tfidf
		}
		logit += classBias
		scores[class] = logit
	}
	return
}

func FindMax(vals []float64) (maxInd int, maxVal float64) {

	maxInd = 0
	maxVal = vals[0]
	for index, value := range vals {
		if value > maxVal {
			maxInd = index
			maxVal = value
		}
	}
	return
}

func (m *Model) SoftMax(logits []float64) (probs []float64) {

	_, maxLogit := FindMax(logits)
	stableLogs := make([]float64, len(logits))
	for score := range logits {
		stableLogs[score] = logits[score] - maxLogit
	}

	probs = make([]float64, len(stableLogs))
	denom := 0.0
	for _, logit := range stableLogs {
		denom += math.Exp(logit)
	}

	for i, logit := range stableLogs {
		probs[i] = math.Exp(logit) / denom
	}
	return
}

func (m *Model) PredictClass(d Document) (p Prediction) {
	p.TF_IDF = m.Vocab.VectorizeDoc(d)
	p.TrueClass = d.Category - 1
	p.RawScores = m.CalcLogits(p.TF_IDF)
	p.Probs = m.SoftMax(p.RawScores)
	p.PredClass, _ = FindMax(p.Probs)

	return
}

// Training Pipeline
func (m *Model) LossGradients(p Prediction) (weightGrads [][]float64, biasGrads []float64) {

	weightGrads = make([][]float64, len(p.Probs))
	biasGrads = make([]float64, len(p.Probs))
	for class := range p.Probs {
		y := 0.0
		weightGrads[class] = make([]float64, m.Vocab.NumWords)
		if class == p.TrueClass {
			y = 1.0
		}
		for vocabIndex, tf_idf := range p.TF_IDF {
			weightGrads[class][vocabIndex] = (p.Probs[class] - y) * tf_idf
		}
		biasGrads[class] = (p.Probs[class] - y)
	}
	return
}

func (m *Model) UpdateWeights(weightGrads [][]float64, biasGrads []float64) {

	for classIndex := range len(m.Classes) {
		for wordIndex, grad := range weightGrads[classIndex] {
			m.Weights[classIndex][wordIndex] -= m.LearnRate * grad
		}
		m.Biases[classIndex] -= m.LearnRate * biasGrads[class]
	}
}

func (m *Model) StocTrain(Docs []Document, epochs int) {

	for round := 0; round < epochs; round++ {
		ShuffelDataset(Docs)
		for _, doc := range Docs {
			prediction := m.PredictClass(doc)
			weights, biases := m.LossGradients(prediction)
			m.UpdateWeights(weights, biases)
		}
	}
}

func (m *Model) BatchUpdate(batch []Document) {

	weights := make([][]float64, len(m.Classes))
	for class := range weights {
		weights[class] = make([]float64, m.Vocab.NumWords)
	}
	biases := make([]float64, len(m.Biases))

	for _, doc := range batch {
		prediction := m.PredictClass(doc)
		weightGrads, biasGrads := m.LossGradients(prediction)

		for classIndex := range weightGrads {
			for wordIndex, grad := range weightGrads[classIndex] {
				weights[classIndex][wordIndex] += grad
			}
			biases[classIndex] += biasGrads[classIndex]
		}
	}
	m.UpdateWeights(weights, biases)
}

func (m *Model) BatchTrain(Docs []Document, batchSize int, epochs int) {

	for round := 0; round < epochs; round++ {
		ShuffelDataset(Docs)
		offset := m.Vocab.NumDocs % batchSize
		if offset != 0 {
			remainder := Docs[m.Vocab.NumDocs-offset:]
			m.BatchUpdate(remainder)

		}
		batchPool := Docs[0 : m.Vocab.NumDocs-offset]
		for current := 0; current < len(batchPool); current += batchSize {
			batch := Docs[current : current+batchSize]
			m.BatchUpdate(batch)
		}
	}
}

func (m *Model) MeasureAccuracy(Docs []Document) {

	numPredictions := 0
	correct := 0
	for _, Doc := range Docs {
		p := m.PredictClass(Doc)
		numPredictions++
		if p.PredClass == p.TrueClass {
			correct++
		}
	}
	m.Accuracy = float64(correct) / float64(numPredictions)

}

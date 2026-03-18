package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

const (
	class = iota
	title
	desc
)

type Document struct {
	Category int
	Tokens   []string
	TFMap    map[string]float64
}

func ParseCSV(filename string) ([][]string, error) {

	file, err := os.Open(filename)
	if err != nil {
		fmt.Printf("Error opening file: %v", err)
		return nil, err
	}

	defer file.Close()

	csvReader := csv.NewReader(file)
	rawDocs, err := csvReader.ReadAll()
	if err != nil {
		fmt.Printf("Error Reading CSV: %v", err)
		return nil, err
	}
	return rawDocs, err
}

func CleanString(s string) string {
	reg, _ := regexp.Compile(`[^a-zA-Z0-9\s\']+`)
	return reg.ReplaceAllString(s, " ")
}

func CreateDocument(doc []string) Document {

	classification, err := strconv.Atoi(doc[class])
	if err != nil {
		classification = 10
	}
	tf := make(map[string]float64)
	tokens := []string{}

	doc[title] = strings.ToLower(doc[title])
	doc[desc] = strings.ToLower(doc[desc])
	contense := CleanString(doc[title] + " " + doc[desc])
	for _, token := range strings.Fields(contense) {
		if token == " " || len(token) == 1 {
			continue
		}

		tokens = append(tokens, token)
		val, ok := tf[token]
		if !ok {
			tf[token] = 1
		} else {
			tf[token] = val + 1
		}
	}

	for word, val := range tf {
		tf[word] = val / float64(len(tokens))
	}

	return Document{
		Category: classification,
		Tokens:   tokens,
		TFMap:    tf,
	}
}

func TokenizeDataset(path string) (docs []Document) {

	rawDocs, err := ParseCSV(path)
	if err != nil {
		fmt.Println(err)
		return
	}
	for _, doc := range rawDocs {
		document := CreateDocument(doc)
		docs = append(docs, document)
	}
	return
}

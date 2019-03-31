package persist_test

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"github.com/goml/gobrain/persist"
)

const value string = `{"foo": "bar"}`

var storingFile = "./persiststorefile_tobedeleted"

func ExampleSave() {
	defer os.Remove(storingFile)

	persist.Save(storingFile, value)

	fileRead, _ := ioutil.ReadFile(storingFile)
	str := string(fileRead)
	fmt.Printf(str)
	// Output:
	// "{\"foo\": \"bar\"}"
}

func ExampleLoad() {
	defer os.Remove(storingFile)

	ioutil.WriteFile(storingFile, []byte(value), 0666)

	var fileLoaded interface{}
	err := persist.Load(storingFile, &fileLoaded)
	if err != nil {
		log.Println(err.Error())
	}
	fmt.Println(fileLoaded)
	// Output:
	// map[foo:bar]
}

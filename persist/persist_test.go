package persist

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
)

const value string = `{"foo": "bar"}`

var storingFile = "./persiststorefile_tobedeleted"

func ExampleSave() {
	defer os.Remove(storingFile)

	Save(storingFile, value)

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
	err := Load(storingFile, &fileLoaded)
	if err != nil {
		log.Println(err.Error())
	}
	fmt.Println(fileLoaded)
	// Output:
	// map[foo:bar]
}

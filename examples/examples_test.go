package examples

func ExampleSimple() {
	Simple()

	// Output:
	// [0.09740879532462123]
}

func ExampleLoad() {
	Load("ff.network")

	// Output:
	// [0.09740879532462095]
}

func ExampleSave() {
	filename := "_saved.network"
	Save(filename)
	Load(filename)

	// Output:
	// [0.09740879532462123]
	// [0.09740879532462123]
}
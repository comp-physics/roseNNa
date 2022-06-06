program name

    USE model
    USE readTester
    implicit none
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: inputs
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: hidden_state
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: cell_state
    REAL, ALLOCATABLE, DIMENSION(:,:,:) :: output
    ALLOCATE(inputs(    1, 2, 5))
    ALLOCATE(hidden_state(    1, 1, 2))
    ALLOCATE(cell_state(    1, 1, 2))
    inputs = RESHAPE(    (/1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0/),    (/1, 2, 5/), order =     [3 , 2 , 1 ])
    hidden_state = RESHAPE(    (/1.0, 1.0/),    (/1, 1, 2/), order =     [3 , 2 , 1 ])
    cell_state = RESHAPE(    (/1.0, 1.0/),    (/1, 1, 2/), order =     [3 , 2 , 1 ])
    CALL initialize()
    print *, "Model Reconstruction Success!"
    CALL use_model(inputs, hidden_state, cell_state, output)
    print *, output
    open(1, file = "goldenFiles/test.txt")
    WRITE(1, *) SHAPE(output)
    output = RESHAPE(output,(/SIZE(output, dim = 3), SIZE(output, dim = 2), SIZE(output, dim = 1)/), order = [3, 2, 1])
    WRITE(1, *) PACK(output,.true.)
end program name
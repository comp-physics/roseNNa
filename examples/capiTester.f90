program name

    USE rosenna
    implicit none
    REAL (c_double), DIMENSION(1,2) :: inputs
    REAL (c_double), DIMENSION(    1, 3) :: output

    inputs = RESHAPE(    (/1.0, 1.0/),    (/1, 2/), order =     [2 , 1 ])

    CALL initialize()

    CALL use_model(inputs, output)

    open(1, file = "test.txt")
    WRITE(1, *) SHAPE(output)
    WRITE(1, *) PACK(RESHAPE(output,(/SIZE(output, dim = 2), SIZE(output, dim = 1)/), order = [2, 1]),.true.)
    print *, output

end program name

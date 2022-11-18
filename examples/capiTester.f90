program name

    USE rosenna
    implicit none
    REAL, DIMENSION(1,2) :: inputs
    REAL, DIMENSION(    1, 3) :: output

    inputs = RESHAPE(    (/1.0, 1.0/),    (/1, 2/), order =     [2 , 1 ])

    CALL initialize()

    CALL use_model(inputs, output)

    print *, output

end program name

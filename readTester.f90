program readTester

    USE derived_types
    USE activation_functions

    implicit none


    TYPE(linLayer), ALLOCATABLE, DIMENSION(:) :: layers

    REAL, ALLOCATABLE, DIMENSION(:,:) :: weights
    INTEGER :: w_dim1
    INTEGER :: w_dim2

    REAL, ALLOCATABLE, DIMENSION(:) :: biases

    INTEGER :: activation_func

    INTEGER :: numLayers
    INTEGER :: i

    open(10, file = "model.txt")
    open(11, file = "weights_biases.txt")

    read(10, *) numLayers
    ALLOCATE(layers(numLayers))
    
    DO i = 1, numLayers
        read(10, *) w_dim1, w_dim2
        ALLOCATE(weights(w_dim1,w_dim2))
        read(11, *) weights

        ALLOCATE(biases(w_dim1))
        read(11, *) biases

        read(10, *) activation_func

        if (activation_func == 0) then
            layers(i)%fn_ptr => relu
        else if (activation_func == 1) then
            layers(i)%fn_ptr => sigmoid
        end if

        layers(i)%weights = weights
        layers(i)%biases = biases

        DEALLOCATE(weights)
        DEALLOCATE(biases)
    END DO

    print *, SHAPE(layers(2)%fn_ptr)


        
end program readTester


PROGRAM linear
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    USE model_layers
    ! ===============================================================

    
    IMPLICIT NONE


    ! INTERFACE
    !     FUNCTION linear_layer(inp, weights, bias) result(output)
    !         real :: output(size(weights, dim=1))
    !         real, intent(in) :: weights(:,:)
    !         real, intent(in) :: inp(:)
    !         real, intent(in) :: bias(:)
    !     END FUNCTION linear_layer
    ! END INTERFACE
    
    REAL :: inp(2)
    REAL, ALLOCATABLE :: out(:)
    INTEGER :: i
    REAL :: T1, T2
    inp = reshape ((/ 1.0 , 1.0  /), shape(inp))

    
    
    ALLOCATE(out(10))
    CALL CPU_TIME(T1)
    CALL init()
    out = inp
    DO i = 1, SIZE(layers)
        out = linear_layer(out, layers(i)%weights, layers(i)%biases)
        out = layers(i)%fn_ptr(out)
    END DO
    call CPU_TIME(T2)

    print *, out
    print *, T2-T1
    
    

END PROGRAM linear

!=======================================================================================

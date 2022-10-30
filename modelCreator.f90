module model
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    ! USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    USE model_layers
    USE readTester
    ! ===============================================================

    
    IMPLICIT NONE
    contains
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
        
        SUBROUTINE use_model(        i0, i1, i2,         o0)
            IMPLICIT NONE
            !INPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 2, 5) :: i0
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 1, 2) :: i1
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 1, 2) :: i2
            !===========================

            !INPUTS CORRESPONDING TO INTERMEDIARY PROCESSING
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:,:,:) :: output0
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:,:) :: output1
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:) :: output2
            !================================================

            !OUTPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(OUT), DIMENSION(        2, 1) :: o0
            !===========================

            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: input
            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: hidden_state
            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: cell_state

            REAL :: T1, T2

            ALLOCATE(input(        1, 2, 5))
            ALLOCATE(hidden_state(        1, 1, 2))
            ALLOCATE(cell_state(        1, 1, 2))

            input = i0
            hidden_state = i1
            cell_state = i2

            
            CALL CPU_TIME(T1)
            
            !========Transpose============
            input = RESHAPE(input,(/SIZE(input, dim = 2), SIZE(input, dim = 1), SIZE(input, dim = 3)/), order = [2, 1, 3])

            !========LSTM Layer============
            CALL lstm(input, hidden_state, cell_state, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh,&
                & output0)

            !========Squeeze============
            output1 = RESHAPE(output0,(/SIZE(output0, dim = 1), SIZE(output0, dim = 3), SIZE(output0, dim = 4)/))

            !========Transpose============
            output1 = RESHAPE(output1,(/SIZE(output1, dim = 2), SIZE(output1, dim = 1), SIZE(output1, dim = 3)/), order = [2, 1, 3])

            !========Reshape============
            output1 = RESHAPE(output1,(/SIZE(output1, dim = 3), SIZE(output1, dim = 2), SIZE(output1, dim = 1)/), order = [3, 2, 1])
            output2 = RESHAPE(output1,(/2, 2/), order = [2, 1])

            !========Gemm Layer============
            CALL linear_layer(output2, linLayers(1),0)
            output2 = relu2d(output2)
            
            !========Gemm Layer============
            CALL linear_layer(output2, linLayers(2),0)
            output2 = sigmoid2d(output2)

            !========Gemm Layer============
            CALL linear_layer(output2, linLayers(3),0)
            output2 = relu2d(output2)
            
            !========Gemm Layer============
            CALL linear_layer(output2, linLayers(4),0)
            output2 = sigmoid2d(output2)

            call CPU_TIME(T2)
            o0 = output2
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================

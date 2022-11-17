module model
    !---------------
    ! adding any number of layers to our neural network
    !----------------


    ! ===============================================================
    ! USE filereader !<loading in weights, biases
    USE activation_functions !<getting activation functions
    USE model_layers
    USE reader
    USE iso_c_binding
    ! ===============================================================

    
    IMPLICIT NONE
    contains
    !===============================================THIS BLOCK COMES FROM FYPP OUTPUT "openNP.fpp"=======================================================
        
        SUBROUTINE use_model(        i0,         o0) bind(c,name="use_model")
            IMPLICIT NONE
            !INPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(INOUT), DIMENSION(        1, 25, 9) :: i0
            !===========================

            !INPUTS CORRESPONDING TO INTERMEDIARY PROCESSING
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:,:,:) :: output0
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:,:) :: output1
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:,:) :: output2
            REAL (c_double), ALLOCATABLE,  DIMENSION(:,:) :: output3
            !================================================

            !OUTPUTS CORRESPONDING TO C
            REAL (c_double), INTENT(OUT), DIMENSION(        1, 9) :: o0
            !===========================

            REAL (c_double), ALLOCATABLE, DIMENSION(:,:,:) :: lstm_1_input

            REAL :: T1, T2

            ALLOCATE(lstm_1_input(        1, 25, 9))

            lstm_1_input = i0

            
            CALL CPU_TIME(T1)
            
            !========Transpose============
            lstm_1_input = RESHAPE(lstm_1_input,(/SIZE(lstm_1_input, dim = 2), SIZE(lstm_1_input, dim = 1), SIZE(lstm_1_input, dim&
                & = 3)/), order = [2, 1, 3])

            !========LSTM Layer============
            output1 = lstmLayers(1)%hid
            output2 = lstmLayers(1)%cell
            CALL lstm(lstm_1_input, output1, output2, lstmLayers(1)%whh, lstmLayers(1)%wih, lstmLayers(1)%bih, lstmLayers(1)%bhh,&
                & output0)

            !========Squeeze============
            output3 = RESHAPE(output1,(/SIZE(output1, dim = 2), SIZE(output1, dim = 3)/))

            !========Gemm Layer============
            CALL linear_layer(output3, linLayers(1),0)

            !===========Add============
            output3 = output3 + RESHAPE(broadc(addLayers(1)%adder,        (/1, 1, 0, 9/),RESHAPE(        (/3, 0/),        (/1,&
                & 2/), order=[2,1])),         (/0, 9/))

            output3 = tanhh2d(output3)

            call CPU_TIME(T2)
            o0 = output3
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================

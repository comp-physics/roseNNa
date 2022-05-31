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
        SUBROUTINE use_model(input, hidden_state, cell_state, output0, output1, output)
            IMPLICIT NONE
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:) :: input
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:) :: hidden_state
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:) :: cell_state
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: output0
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:) :: output1
            REAL, ALLOCATABLE, INTENT(OUT), DIMENSION(:,:,:) :: output
            REAL :: T1, T2
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

            call CPU_TIME(T2)
            print *, "-------------"
            output = output1
            print *, "TIME TAKEN:", T2-T1
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================

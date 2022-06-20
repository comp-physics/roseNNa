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
        SUBROUTINE use_model(input, output)
            IMPLICIT NONE
            REAL, ALLOCATABLE, INTENT(INOUT), DIMENSION(:,:,:,:) :: input
            REAL, ALLOCATABLE, INTENT(OUT), DIMENSION(:,:,:,:) :: output
            REAL :: T1, T2
            CALL CPU_TIME(T1)
            
            !========Conv Layer============
            CALL conv(input, convLayers(1)%weights, convLayers(1)%biases,         (/1, 1/),         (/0, 0, 0, 0/),         (/1,&
                & 1/))

            call CPU_TIME(T2)
            print *, "-------------"
            output = input
            print *, "TIME TAKEN:", T2-T1
        end SUBROUTINE
    !===================================================================================================================================================
        

END module model

!=======================================================================================

module FILEREADER
    ! defining derived types ===================================================
    USE derived_types
    USE activation_functions
    ! ==========================================================================


    implicit none

    TYPE(linLayer), ALLOCATABLE, DIMENSION(:) :: layers
    
    ! saved old code =============================================================================================================================
    !> ${"weights_0, bias_0"}$#{for index in range(int(len(statedict)/2-1))}#${", "}$weights_${index+1}$${", "}$bias_${index+1}$#{endfor}#
    ! ============================================================================================================================================

    contains

    subroutine init()
        REAL, ALLOCATABLE, DIMENSION(:,:) :: weights
        INTEGER :: w_dim1
        INTEGER :: w_dim2

        REAL, ALLOCATABLE, DIMENSION(:) :: biases

        INTEGER :: activation_func

        INTEGER :: numLayers
        INTEGER :: i
        CHARACTER(LEN = 10) :: layerName
        open(10, file = "model.txt")
        open(11, file = "weights_biases.txt")

        read(10, *) numLayers
        ALLOCATE(layers(numLayers))
        
        DO i = 1, numLayers
            read(10, *) layerName
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
    end subroutine
end module

using Distributions
using Plots
using LaTeXStrings
using Random
using Statistics

script_dir = @__DIR__
code_dir = normpath(joinpath(script_dir, ".."))
csv_dir = joinpath(code_dir, "csv_files")
mkpath(csv_dir)

function generate_observations(gamma_dB; NMC=500_000, Ka=300, J=12, seed=2026)
    Random.seed!(seed)

    Z = randn(NMC)

    q = 2.0^(-J)

    k_vec = collect(1:Ka)
    bin_dist = Binomial(Ka, q)
    prob_k_vec = pdf.(bin_dist, k_vec)

    p0 = pdf(bin_dist, 0)
    p_active = 1 - p0
    prob_k_given_active = prob_k_vec ./ p_active

    active_count_dist = Categorical(prob_k_given_active)
    S_samples = rand(active_count_dist, NMC)

    p_s_eq_1_empirical = mean(S_samples .== 1)
    p_s_ge_2_empirical = mean(S_samples .>= 2)

    gamma_lin = 10.0^(gamma_dB / 10)

    R_h1 = sqrt(gamma_lin) .* S_samples .+ Z
    R_h0 = randn(NMC)

    return R_h1, R_h0, p_s_eq_1_empirical, p_s_ge_2_empirical
end

function sweep_thresholds(; gamma_dB=10.0, NMC=500_000, Ka=300, J=12, thresholds=collect(range(-5.0, 10.0, length=50)), seed=2026)
    R_h1, R_h0, p_s_eq_1_empirical, p_s_ge_2_empirical = generate_observations(
        gamma_dB;
        NMC=NMC,
        Ka=Ka,
        J=J,
        seed=seed,
    )

    p_fa_empirical = zeros(Float64, length(thresholds))
    p_md_empirical = zeros(Float64, length(thresholds))

    for (idx, a) in enumerate(thresholds)
        p_fa_empirical[idx] = mean(R_h0 .>= a)
        p_md_empirical[idx] = mean(R_h1 .< a)
        println("a = $(round(a, digits=3)), p_fa_empirical = $(p_fa_empirical[idx]), p_md_empirical = $(p_md_empirical[idx])")
    end

    return thresholds, p_fa_empirical, p_md_empirical, p_s_eq_1_empirical, p_s_ge_2_empirical
end

NMC = parse(Int, get(ENV, "NMC", "20000000"))
gamma_dB_values = [10.0, 13.0, 15.0]
thresholds = collect(range(-2.0, 5.0, length=50))

mc_csv = joinpath(csv_dir, "fig2_monte_carlo.csv")
open(mc_csv, "w") do io
    println(io, "snr_dB,threshold,p_fa,p_md,p_s_eq_1,p_s_ge_2,NMC")

    for gamma_dB in gamma_dB_values
        println("Sweeping gamma_dB = $(gamma_dB)")
        thresholds_swept, p_fa_empirical, p_md_empirical, p_s_eq_1_empirical, p_s_ge_2_empirical = sweep_thresholds(
            ; gamma_dB=gamma_dB,
            NMC=NMC,
            thresholds=thresholds,
        )

        for idx in eachindex(thresholds_swept)
            println(io, "$(gamma_dB),$(thresholds_swept[idx]),$(p_fa_empirical[idx]),$(p_md_empirical[idx]),$(p_s_eq_1_empirical),$(p_s_ge_2_empirical),$(NMC)")
        end
    end
end

println("Wrote Monte Carlo results to $(mc_csv)")

using Distributions
using Plots
using LaTeXStrings
using Random

script_dir = @__DIR__
code_dir = normpath(joinpath(script_dir, ".."))
csv_dir = joinpath(code_dir, "csv_files")
mkpath(csv_dir)

effective_snr_dB = [10.0, 13.0, 15.0]

# P_FA values from 10^-6 to 10^-1
p_fa = 10.0 .^ range(-6, -1, length=1000)

standard_normal = Normal(0, 1)
theoretical_rows = Vector{Tuple{Float64, Float64, Float64}}()

plt = plot(
    xscale=:log10,
    yscale=:log10,
    xlabel=L"P_{\mathrm{FA}}",
    ylabel=L"P_{\mathrm{MD}}",
    legend=:best,

    grid=true,
    minorgrid=true,
    minorticks=10,

    xlims=(1e-6, 1e-1),
    ylims=(1e-6, 1.0)
)

for snr_dB in effective_snr_dB
    γ = 10.0^(snr_dB / 10.0)

    # Q⁻¹(P_FA)
    q_inv_pfa = quantile.(standard_normal, 1.0 .- p_fa)

    # P_MD = Q(sqrt(γ) - Q⁻¹(P_FA))
    p_md = ccdf.(
        standard_normal,
        sqrt(γ) .- q_inv_pfa
    )

    append!(theoretical_rows, zip(fill(snr_dB, length(p_fa)), p_fa, p_md))

    plot!(
        plt,
        p_fa,
        p_md,
        linestyle=:dash,
        linewidth=2,
        label="$(Int(round(snr_dB))) dB"
    )
end

theoretical_csv = joinpath(csv_dir, "fig2_theoretical.csv")
open(theoretical_csv, "w") do io
    println(io, "snr_dB,p_fa,p_md")
    for (snr_dB, pfa, pmd) in theoretical_rows
        println(io, "$(snr_dB),$(pfa),$(pmd)")
    end
end

println("Wrote theoretical results to $(theoretical_csv)")

display(plt)


